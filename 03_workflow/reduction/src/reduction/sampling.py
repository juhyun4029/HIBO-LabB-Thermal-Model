"""Independent-chain Metropolis sampling for merged-block parameters.

R/C use log coordinates; merged/air gain fractions use logit coordinates.
Four independent PCG64 streams run in threads, with the physical target's
Numba kernel releasing the GIL. Adaptation is confined to warmup. Fixed-kernel
settling draws are discarded and all production draws are retained.
Checkpoints use lossless NPZ/JSON. Convergence is assessed empirically."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


SAMPLER_SCHEMA = "hibo-reduced-block-mixed-mh-v2"
DEFAULT_CONFIG = {
    "chains": 4,
    "warmup": 12000,
    "settling": 12000,
    "min_draws": 6000,
    "max_draws": 24000,
    "check_every": 1000,
    "checkpoint_every": 250,
    "target_accept": 0.234,
    "coordinate_target_accept": 0.44,
    "block_probability": 0.7,
    "initial_step": 0.25,
    "initial_jitter": 1.0,
    "adapt_start": 100,
    "adapt_every": 25,
    "rhat_threshold": 1.10,
    "bulk_ess_min": 400,
    "tail_ess_min": 400,
    "smoke": False,
}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _json(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _replace_with_retry(source, destination):
    """Bounded retry for transient Windows file locks; never delete old state.

    If all attempts fail, the previous destination is left intact and the
    fully written temporary file remains available for inspection. No other
    exception is swallowed. Total scheduled waiting is 1.55 seconds.
    """
    delays = (0.05, 0.10, 0.20, 0.40, 0.80)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(tmp, path)


def _configuration(config):
    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown sampler configuration: {sorted(unknown)}")
    result = dict(DEFAULT_CONFIG, **supplied)
    integer_keys = ("chains", "warmup", "settling", "min_draws", "max_draws",
                    "check_every", "checkpoint_every", "adapt_start", "adapt_every")
    for key in integer_keys:
        if isinstance(result[key], bool) or int(result[key]) != result[key]:
            raise ValueError(f"{key} must be an integer")
        result[key] = int(result[key])
    if result["chains"] != 4:
        raise ValueError("This study requires four independent chains")
    if result["warmup"] < 0 or result["settling"] < 0:
        raise ValueError("Warmup and settling counts cannot be negative")
    for key in ("min_draws", "max_draws", "check_every", "checkpoint_every", "adapt_start", "adapt_every"):
        if result[key] < 1:
            raise ValueError(f"{key} must be positive")
    if result["min_draws"] < 4 or result["max_draws"] < result["min_draws"]:
        raise ValueError("Require 4 <= min_draws <= max_draws")
    for key in ("target_accept", "coordinate_target_accept", "block_probability"):
        if not 0 < result[key] < 1:
            raise ValueError(f"{key} must lie strictly between zero and one")
    for key in ("initial_step", "initial_jitter", "bulk_ess_min", "tail_ess_min"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 1 < result["rhat_threshold"] <= 1.10:
        raise ValueError("R-hat threshold must be > 1 and no looser than 1.10")
    if result["bulk_ess_min"] < 400 or result["tail_ess_min"] < 400:
        raise ValueError("The study ESS gates cannot be relaxed below 400")
    if not isinstance(result["smoke"], bool):
        raise ValueError("smoke must be a boolean")
    return result


def _digest_object(value, digest):
    """Stable structural hash, without serializing large arrays as JSON."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            _digest_object(array.tolist(), digest)
            return
        digest.update(b"array:")
        digest.update(str(array.dtype).encode())
        digest.update(_json(array.shape).encode())
        digest.update(memoryview(array).cast("B"))
    elif isinstance(value, dict):
        digest.update(b"dict:")
        for key in sorted(value, key=str):
            _digest_object(str(key), digest)
            _digest_object(value[key], digest)
    elif isinstance(value, (tuple, list)):
        digest.update(b"list:")
        for entry in value:
            _digest_object(entry, digest)
    else:
        digest.update(_json(value).encode())
        digest.update(b";")


def _target_fingerprint(target):
    supplied = getattr(target, "fingerprint", None)
    if supplied is not None:
        value = supplied() if callable(supplied) else supplied
        if value is None:
            raise ValueError("Target fingerprint cannot be None")
        return str(value)
    content = {"class": type(target).__module__ + "." + type(target).__qualname__}
    for key in ("model", "free_specs", "data", "likelihood_minutes", "initial_log_values"):
        value = getattr(target, key, None)
        if value is not None and not callable(value):
            content[key] = value
    if len(content) == 1:
        raise ValueError("Target must expose a stable fingerprint or model/free_specs/data")
    digest = hashlib.sha256()
    _digest_object(content, digest)
    return digest.hexdigest()


def _initial_coordinates(target):
    value = getattr(target, "initial_log_values", None)
    if value is not None:
        initial = np.asarray(value() if callable(value) else value, dtype=float)
    else:
        specs = getattr(target, "free_specs", None)
        if not specs:
            raise ValueError("Target must expose free_specs or initial_log_values")
        coordinates = []
        for spec in specs:
            value = float(spec["initial"])
            transform = spec.get("transform", "log")
            if transform == "log":
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("Initial R/C parameters must be strictly positive")
                coordinates.append(math.log(value))
            elif transform == "logit":
                budget = float(spec["budget"])
                if not np.isfinite(budget) or not 0 < value < budget:
                    raise ValueError("Initial alpha must be strictly inside its allocation budget")
                coordinates.append(math.log(value) - math.log(budget - value))
            else:
                raise ValueError(f"Unknown coordinate transform: {transform}")
        initial = np.asarray(coordinates, dtype=float)
    if initial.ndim != 1 or not initial.size or not np.isfinite(initial).all():
        raise ValueError("Initial coordinates must be a finite nonempty vector")
    return initial


def _physical_coordinates(target, coordinates):
    """Use the target's actual transformation, including alpha budgets."""
    coordinates = np.asarray(coordinates, dtype=float)
    transform = getattr(target, "physical_values", None)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        physical = np.asarray(transform(coordinates) if callable(transform)
                              else np.exp(coordinates), dtype=float)
    if physical.shape != coordinates.shape:
        raise ValueError("Target physical_values must preserve the coordinate array shape")
    return physical


def _finite_logp(target, position):
    value = float(target.logp(position))
    return value if np.isfinite(value) else -math.inf


def _new_state(target, initial, child_seed, config):
    rng = np.random.Generator(np.random.PCG64(child_seed))
    # Independently dispersed starts; finite-state rejection never changes the target.
    for _ in range(100):
        position = initial + config["initial_jitter"] * rng.normal(size=initial.size)
        logp = _finite_logp(target, position)
        if np.isfinite(logp):
            break
    else:
        raise ValueError("Unable to initialize a finite dispersed chain after 100 attempts")
    covariance = np.eye(initial.size) * config["initial_step"] ** 2
    return {
        "position": position, "logp": logp, "rng": rng,
        "warmup_completed": 0, "settling_completed": 0, "production_completed": 0,
        "adapt_count": 1, "adapt_mean": position.copy(),
        "adapt_m2": np.zeros((initial.size, initial.size)),
        "covariance": covariance, "chol": np.linalg.cholesky(covariance),
        "log_scale": 0.0,
        "block_adapt_count": 0,
        "coordinate_log_scales": np.zeros(initial.size),
        "coordinate_adapt_counts": np.zeros(initial.size, dtype=np.int64),
        "accepted": {"warmup": 0, "settling": 0, "production": 0},
        "draws": [], "logps": [],
    }


def _save_state(path, state, fingerprint):
    metadata = {key: state[key] for key in ("logp", "warmup_completed", "settling_completed",
                "production_completed", "adapt_count", "log_scale", "block_adapt_count", "accepted")}
    metadata.update(schema=SAMPLER_SCHEMA, fingerprint=fingerprint,
                    rng_state=state["rng"].bit_generator.state)
    dim = len(state["position"])
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    with open(tmp, "wb") as stream:
        np.savez(stream, metadata=np.array(_json(metadata)), position=state["position"],
                 adapt_mean=state["adapt_mean"], adapt_m2=state["adapt_m2"],
                 covariance=state["covariance"], chol=state["chol"],
                 coordinate_log_scales=state["coordinate_log_scales"],
                 coordinate_adapt_counts=state["coordinate_adapt_counts"],
                 draws=np.asarray(state["draws"], dtype=float).reshape(-1, dim),
                 logps=np.asarray(state["logps"], dtype=float))
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(tmp, path)


def _load_state(path, fingerprint, config, dimension):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if metadata.get("schema") != SAMPLER_SCHEMA or metadata.get("fingerprint") != fingerprint:
            raise ValueError(f"Incompatible chain checkpoint: {path}")
        state = {key: saved[key].copy() for key in ("position", "adapt_mean", "adapt_m2", "covariance", "chol",
                                                   "coordinate_log_scales", "coordinate_adapt_counts")}
        draws = saved["draws"].copy()
        logps = saved["logps"].copy()
    for key in ("logp", "warmup_completed", "settling_completed", "production_completed", "adapt_count", "log_scale", "block_adapt_count", "accepted"):
        state[key] = metadata[key]
    if state["position"].shape != (dimension,) or draws.shape != (state["production_completed"], dimension):
        raise ValueError(f"Checkpoint shape mismatch: {path}")
    if logps.shape != (state["production_completed"],):
        raise ValueError(f"Checkpoint log-posterior length mismatch: {path}")
    if state["coordinate_log_scales"].shape != (dimension,) or state["coordinate_adapt_counts"].shape != (dimension,):
        raise ValueError(f"Checkpoint coordinate-adaptation shape mismatch: {path}")
    if np.any(state["coordinate_adapt_counts"] < 0) or not np.issubdtype(state["coordinate_adapt_counts"].dtype, np.integer):
        raise ValueError("Checkpoint coordinate adaptation counts must be nonnegative integers")
    if not isinstance(state["block_adapt_count"], int) or state["block_adapt_count"] < 0:
        raise ValueError("Checkpoint block adaptation count must be nonnegative")
    if state["block_adapt_count"] + int(state["coordinate_adapt_counts"].sum()) != state["warmup_completed"]:
        raise ValueError("Checkpoint proposal adaptation counts disagree with warmup counter")
    for phase, limit in (("warmup", config["warmup"]), ("settling", config["settling"]), ("production", config["max_draws"])):
        count = state[phase + "_completed"]
        if not isinstance(count, int) or not 0 <= count <= limit:
            raise ValueError(f"Checkpoint has invalid {phase} counter: {path}")
    if state["settling_completed"] and state["warmup_completed"] != config["warmup"]:
        raise ValueError("Checkpoint entered settling before warmup finished")
    if state["production_completed"] and state["settling_completed"] != config["settling"]:
        raise ValueError("Checkpoint entered production before settling finished")
    if not all(np.isfinite(state[key]).all() for key in ("position", "adapt_mean", "adapt_m2", "covariance", "chol", "coordinate_log_scales", "log_scale")):
        raise ValueError("Checkpoint contains nonfinite sampler state")
    if not np.isfinite(draws).all() or not np.isfinite(logps).all() or not np.isfinite(state["logp"]):
        raise ValueError("Checkpoint contains nonfinite retained states")
    state["rng"] = np.random.Generator(np.random.PCG64())
    state["rng"].bit_generator.state = metadata["rng_state"]
    state["draws"] = list(draws)
    state["logps"] = list(logps)
    return state


def _adapt(state, acceptance_probability, config, coordinate=None):
    """Warmup-only covariance adaptation plus proposal-specific scales."""
    state["adapt_count"] += 1
    n = state["adapt_count"]
    delta = state["position"] - state["adapt_mean"]
    state["adapt_mean"] += delta / n
    state["adapt_m2"] += np.outer(delta, state["position"] - state["adapt_mean"])
    iteration = state["warmup_completed"]
    if coordinate is None:
        state["block_adapt_count"] += 1
        rate = (state["block_adapt_count"] + 10.0) ** -0.6
        state["log_scale"] = float(np.clip(state["log_scale"] + rate * (acceptance_probability - config["target_accept"]), -6.0, 3.0))
    else:
        state["coordinate_adapt_counts"][coordinate] += 1
        rate = (state["coordinate_adapt_counts"][coordinate] + 10.0) ** -0.6
        state["coordinate_log_scales"][coordinate] = np.clip(
            state["coordinate_log_scales"][coordinate]
            + rate * (acceptance_probability - config["coordinate_target_accept"]), -6.0, 3.0)
    if iteration >= config["adapt_start"] and iteration % config["adapt_every"] == 0:
        dimension = state["position"].size
        empirical = state["adapt_m2"] / (n - 1)
        covariance = (n * empirical + 20 * config["initial_step"] ** 2 * np.eye(dimension)) / (n + 20)
        covariance = (covariance + covariance.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.clip(eigenvalues, 1e-6, 1e6)) @ eigenvectors.T
        state["covariance"] = covariance
        state["chol"] = np.linalg.cholesky(covariance)


def _advance(target, state, phase, goal, config, stop_path):
    """Consume only this chain's RNG; chunk boundaries never affect proposals."""
    key = phase + "_completed"
    dimension = len(state["position"])
    while state[key] < goal:
        if stop_path is not None and stop_path.exists():
            break
        rng = state["rng"]
        coordinate = None
        if rng.random() < config["block_probability"]:
            scale = math.exp(state["log_scale"]) * 2.38 / math.sqrt(dimension)
            increment = scale * (state["chol"] @ rng.normal(size=dimension))
        else:
            coordinate = int(rng.integers(dimension))
            # One-dimensional moves use their own warmup-adapted scale and
            # do not inherit the full-block 1/sqrt(dimension) shrinkage.
            scale = math.exp(state["coordinate_log_scales"][coordinate]) * 2.38
            increment = np.zeros(dimension)
            increment[coordinate] = scale * math.sqrt(state["covariance"][coordinate, coordinate]) * rng.normal()
        proposed = state["position"] + increment
        proposed_logp = _finite_logp(target, proposed)
        log_alpha = min(0.0, proposed_logp - state["logp"])
        accepted = math.log(max(float(rng.random()), np.finfo(float).tiny)) < log_alpha
        if accepted:
            state["position"] = proposed
            state["logp"] = proposed_logp
            state["accepted"][phase] += 1
        state[key] += 1
        if phase == "warmup":
            _adapt(state, math.exp(log_alpha), config, coordinate)
        elif phase == "production":
            state["draws"].append(state["position"].copy())
            state["logps"].append(state["logp"])
    return state


def _numeric_or_none(value):
    return float(value) if np.isfinite(value) else None


def _coordinate_diagnostics(values, names, config):
    import arviz as az
    import xarray as xr

    if not np.isfinite(values).all():
        return {"passed": False, "reason": "Nonfinite transformed coordinates", "per_parameter": []}
    dataset = xr.Dataset({"theta": (("chain", "draw", "parameter"), values)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rhat = np.asarray(az.rhat(dataset, method="rank")["theta"])
        bulk = np.asarray(az.ess(dataset, method="bulk", relative=False)["theta"])
        q05 = np.asarray(az.ess(dataset, method="quantile", prob=0.05, relative=False)["theta"])
        q95 = np.asarray(az.ess(dataset, method="quantile", prob=0.95, relative=False)["theta"])
    tail = np.minimum(q05, q95)
    finite = np.isfinite(rhat) & np.isfinite(bulk) & np.isfinite(tail)
    passed = finite & (rhat < config["rhat_threshold"]) & (bulk >= config["bulk_ess_min"]) & (tail >= config["tail_ess_min"])
    return {
        "passed": bool(passed.all()),
        "max_rhat": _numeric_or_none(np.max(rhat)),
        "min_bulk_ess": _numeric_or_none(np.min(bulk)),
        "min_tail_ess": _numeric_or_none(np.min(tail)),
        "per_parameter": [dict(name=name, rhat=_numeric_or_none(r), bulk_ess=_numeric_or_none(b),
                               tail_ess=_numeric_or_none(t), ess_q05=_numeric_or_none(l),
                               ess_q95=_numeric_or_none(u), passed=bool(p))
                          for name, r, b, t, l, u, p in zip(names, rhat, bulk, tail, q05, q95, passed)],
    }


def diagnose(chains, names=None, config=None, physical_values=None):
    """Diagnose unconstrained coordinates and the corresponding physical values.

    ``physical_values`` must be supplied for mixed log/logit coordinates.
    Omission preserves the positive-log-only synthetic-target interface.
    The ``log`` result key is a legacy alias of ``transformed``; the latter
    is the accurate label when alpha logits are included.
    """
    config = _configuration(config)
    values = np.asarray(chains, dtype=float)
    if values.ndim != 3 or values.shape[0] != 4 or values.shape[2] < 1:
        raise ValueError("Expected (4 chains, draws, positive parameter count)")
    names = list(names or [f"parameter_{i + 1}" for i in range(values.shape[2])])
    if len(names) != values.shape[2]:
        raise ValueError("Parameter-name count does not match chain dimension")
    if physical_values is None:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            physical = np.exp(values)
        transformation = "exp fallback for positive-log-only coordinates"
    else:
        physical = np.asarray(physical_values, dtype=float)
        if physical.shape != values.shape:
            raise ValueError("Physical diagnostics must match the chain/draw/parameter shape")
        transformation = "target-provided transformation to actual R/C and merged alpha"
    definition = {"rhat_method": "rank", "bulk_ess_method": "bulk",
                  "tail_ess_method": "minimum of explicit quantile ESS at 0.05 and 0.95",
                  "relative_ess": False, "coordinates": ["transformed", "physical"],
                  "transformation": transformation,
                  "legacy_log_key": "Alias of transformed; includes alpha logit coordinates when sampled",
                  "draw_policy": "All retained production draws; no thinning or posthoc discard",
                  "thresholds": {"rhat_strictly_below": config["rhat_threshold"],
                                 "bulk_ess_at_least": config["bulk_ess_min"], "tail_ess_at_least": config["tail_ess_min"]}}
    if values.shape[1] < 4:
        return {"passed": False, "reason": "Fewer than four retained draws", "definition": definition,
                "transformed": {"passed": False, "per_parameter": []},
                "log": {"passed": False, "per_parameter": []}, "physical": {"passed": False, "per_parameter": []}}
    log_result = _coordinate_diagnostics(values, names, config)
    physical_result = _coordinate_diagnostics(physical, names, config)
    maxima = [r.get("max_rhat") for r in (log_result, physical_result)]
    bulks = [r.get("min_bulk_ess") for r in (log_result, physical_result)]
    tails = [r.get("min_tail_ess") for r in (log_result, physical_result)]
    return {"passed": log_result["passed"] and physical_result["passed"], "definition": definition,
            "max_rhat": max(maxima) if all(x is not None for x in maxima) else None,
            "min_bulk_ess": min(bulks) if all(x is not None for x in bulks) else None,
            "min_tail_ess": min(tails) if all(x is not None for x in tails) else None,
            "transformed": log_result, "log": log_result, "physical": physical_result}


def _runtime_identity():
    packages = {}
    for package in ("numpy", "scipy", "arviz", "arviz-stats", "arviz-base", "xarray", "numba", "llvmlite"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version.split()[0], "machine": platform.machine(), "system": platform.system(), "packages": packages}


def run_chains(target, output_dir, config=None, seed=20261019, resume=True,
               stop_path=None, progress_callback=None):
    """Run or deterministically resume four independent merged-block chains.

    A STOP file is observed during proposals; delete it before resuming. An
    incompatible sampler/target/config/seed/runtime fingerprint is an error.
    A root-level trial lock must prevent concurrent writers to ``output_dir``.
    Smoke runs never receive ``converged`` status, even if a small check passes.
    """
    config = _configuration(config)
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stop_path = Path(stop_path) if stop_path is not None else None
    initial = _initial_coordinates(target)
    specs = getattr(target, "free_specs", None)
    names = [str(item.get("name", f"parameter_{i+1}")) for i, item in enumerate(specs)] if specs else [f"parameter_{i+1}" for i in range(len(initial))]
    if len(names) != initial.size:
        raise ValueError("free_specs and initial coordinates disagree")
    coordinate_transforms = [str(s.get("transform", "log")) for s in specs] if specs else ["log"] * len(initial)
    if "logit" in coordinate_transforms and not callable(getattr(target, "physical_values", None)):
        raise ValueError("A target with alpha logit coordinates must provide physical_values")
    identity = {"schema": SAMPLER_SCHEMA,
                "sampler_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "target": _target_fingerprint(target), "config": config,
                "seed": seed, "initial_unconstrained_values": initial.tolist(),
                "coordinate_transforms": coordinate_transforms, "runtime": _runtime_identity()}
    fingerprint = hashlib.sha256(_json(identity).encode()).hexdigest()
    meta_path = output / "sampler_meta.json"
    checkpoints = [output / f"chain_{i + 1:02d}.npz" for i in range(4)]
    if meta_path.exists():
        if not resume:
            raise FileExistsError(f"Sampler output already exists: {output}; choose a fresh directory")
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise ValueError("Incompatible resume: sampler, target, config, seed, or runtime changed")
    else:
        if any(path.exists() for path in checkpoints):
            raise ValueError("Orphan chain checkpoint without sampler_meta.json")
        _atomic_json(meta_path, {"fingerprint": fingerprint, "identity": identity,
                                "parameter_names": names, "coordinate_transforms": coordinate_transforms})
    seeds = np.random.SeedSequence(seed).spawn(4)
    states = [_load_state(path, fingerprint, config, len(initial)) if path.exists()
              else _new_state(target, initial, seeds[i], config) for i, path in enumerate(checkpoints)]
    for path, state in zip(checkpoints, states):
        _save_state(path, state, fingerprint)
    history_path = output / "diagnostic_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    common_draws = min(s["production_completed"] for s in states)
    # A previous process can stop after saving a check but before status writing.
    history = [entry for entry in history if entry["draws_per_chain"] <= common_draws]

    def arrays():
        common = min(s["production_completed"] for s in states)
        return (np.asarray([s["draws"][:common] for s in states], dtype=float).reshape(4, common, len(initial)),
                np.asarray([s["logps"][:common] for s in states], dtype=float).reshape(4, common))

    def evaluate():
        values, _ = arrays()
        count = values.shape[1]
        if history and history[-1]["draws_per_chain"] == count:
            return history[-1]["diagnostics"]
        info = diagnose(values, names, config, physical_values=_physical_coordinates(target, values))
        history.append({"draws_per_chain": count, "diagnostics": info})
        _atomic_json(history_path, history)
        return info

    def report(phase):
        record = {"phase": phase, "warmup_completed": [s["warmup_completed"] for s in states],
                  "settling_completed": [s["settling_completed"] for s in states],
                  "production_completed": [s["production_completed"] for s in states],
                  "config": config, "latest_diagnostics": history[-1] if history else None}
        _atomic_json(output / "progress.json", record)
        if progress_callback is not None:
            progress_callback(record)

    def finish(status, diagnostics=None):
        values, logps = arrays()
        physical = _physical_coordinates(target, values)
        if diagnostics is None:
            diagnostics = diagnose(values, names, config, physical_values=physical)
        representative, representative_logp, representative_index = None, None, None
        representative_physical = None
        if logps.size:
            index = np.unravel_index(int(np.argmax(logps)), logps.shape)
            representative = values[index].tolist()
            representative_physical = physical[index].tolist()
            representative_logp = float(logps[index])
            representative_index = {"chain_one_based": int(index[0] + 1), "retained_draw_one_based": int(index[1] + 1)}
        result = {"status": status, "chains": values, "physical_chains": physical,
                  "logp": logps, "diagnostics": diagnostics,
                  "history": history, "representative_log_values": representative,
                  "representative_unconstrained_values": representative,
                  "representative_physical_values": representative_physical,
                  "representative_logp": representative_logp, "representative_index": representative_index,
                  "representative_definition": "Retained joint draw with maximum target density in the declared unconstrained coordinates (log R/C, logit alpha fractions); not a physical-coordinate MAP or an optimized/global MAP",
                  "parameter_names": names, "config": config, "fingerprint": fingerprint,
                  "draws_per_chain": values.shape[1], "chain_coordinate": "unconstrained_log_R_C_logit_alpha",
                  "coordinate_transforms": coordinate_transforms,
                  "legacy_key_note": "chains and representative_log_values contain unconstrained coordinates; use physical_chains and representative_physical_values for actual R/C/alpha",
                  "acceptance": [{phase: s["accepted"][phase] / s[phase + "_completed"] if s[phase + "_completed"] else None
                                  for phase in ("warmup", "settling", "production")} for s in states],
                  "reason": "Smoke run cannot authorize adoption" if config["smoke"] else
                            ("All sampler convergence gates passed" if status == "converged" else
                             "Stop requested; retained state is resumable" if status == "stopped" else
                             "Maximum production draws reached without all convergence gates")}
        _atomic_json(output / "sampler_result.json", {k: v for k, v in result.items() if k not in ("chains", "physical_chains", "logp")})
        return result

    if stop_path is not None and stop_path.exists():
        return finish("stopped")
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="hibo-chain") as pool:
        for phase, total in (("warmup", config["warmup"]), ("settling", config["settling"])):
            key = phase + "_completed"
            while min(s[key] for s in states) < total:
                common = min(s[key] for s in states)
                goal = min(total, (common // config["checkpoint_every"] + 1) * config["checkpoint_every"])
                futures = [pool.submit(_advance, target, s, phase, goal, config, stop_path) for s in states]
                for i, future in enumerate(futures):
                    states[i] = future.result()
                    _save_state(checkpoints[i], states[i], fingerprint)
                report(phase)
                if stop_path is not None and stop_path.exists():
                    return finish("stopped")
        due = sorted(set(range(config["min_draws"], config["max_draws"] + 1, config["check_every"])) | {config["max_draws"]})
        for assessment in due:
            if assessment < min(s["production_completed"] for s in states):
                continue
            while min(s["production_completed"] for s in states) < assessment:
                common = min(s["production_completed"] for s in states)
                goal = min(assessment, (common // config["checkpoint_every"] + 1) * config["checkpoint_every"])
                futures = [pool.submit(_advance, target, s, "production", goal, config, stop_path) for s in states]
                for i, future in enumerate(futures):
                    states[i] = future.result()
                    _save_state(checkpoints[i], states[i], fingerprint)
                report("production")
                if stop_path is not None and stop_path.exists():
                    return finish("stopped")
            diagnostics = evaluate()
            report("diagnostics")
            if diagnostics["passed"] and not config["smoke"]:
                return finish("converged", diagnostics)
    return finish("unresolved", evaluate())
