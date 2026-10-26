"""Boundary-preserving RC simulation and conditional reduced-model target.

Temperatures are end-of-minute values. Each minute's forcing is held constant
through two 30-second backward-Euler steps. Every boundary port remains
explicit. The merged block's R/C and source allocations are sampled; source
allocations share a fixed merged-plus-air budget. All other block parameters
and hourly noise parameters are fixed at the adopted parent values.

The conditional target does not propagate fixed-parameter uncertainty.
Independent lognormal priors are specified for aggregate R/C parameters;
they are not exact distributions induced by aggregation. Heat allocations
use a conditional Beta split, with Dirichlet concentrations aggregated by
the number of initial massive components."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

LOG_VARIANCE = float(np.log(5.0))


def load_data(path: str | Path) -> dict[str, np.ndarray]:
    """Load a non-pickled NPZ; preserve the supplied experiment split masks."""
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    return _validated_data(result)


def _validated_data(data: dict[str, Any]) -> dict[str, np.ndarray]:
    d = {str(k): np.array(v, copy=True) for k, v in data.items()}
    required = ("y", "boundaries", "g_hvac", "tsupply", "gains", "ginf", "dt_seconds")
    missing = [name for name in required if name not in d]
    if missing:
        raise ValueError(f"Missing physical input arrays: {missing}")
    n = len(d["y"])
    if n < 1 or d["y"].shape != (n,):
        raise ValueError("y must be a nonempty one-dimensional temperature series")
    if float(d["dt_seconds"]) != 60.0:
        raise ValueError("Physical input interval must be exactly 60 seconds")
    for key in ("y", "g_hvac", "tsupply", "ginf"):
        d[key] = np.ascontiguousarray(d[key], dtype=np.float64)
        if d[key].shape != (n,) or not np.isfinite(d[key]).all():
            raise ValueError(f"{key} must contain {n} finite values")
    for key in ("boundaries", "gains"):
        d[key] = np.ascontiguousarray(d[key], dtype=np.float64)
        if d[key].shape != (n, 4) or not np.isfinite(d[key]).all():
            raise ValueError(f"{key} must have finite shape ({n}, 4)")
    if np.any(d["g_hvac"] < 0) or np.any(d["ginf"] < 0):
        raise ValueError("HVAC and infiltration conductances cannot be negative")
    if "train" not in d or "valid" not in d:
        if "day" not in d:
            raise ValueError("Provide train/valid masks, or the original July day array")
        conditioning = d.get("conditioning", np.arange(n) < 1440).astype(bool)
        d["train"] = (d["day"] <= 22) & ~conditioning
        d["valid"] = (d["day"] >= 23) & ~conditioning
        d["conditioning"] = conditioning
    for key in ("train", "valid", "conditioning"):
        if key not in d:
            d[key] = ~(d["train"].astype(bool) | d["valid"].astype(bool))
        d[key] = np.ascontiguousarray(d[key], dtype=np.bool_)
        if d[key].shape != (n,):
            raise ValueError(f"{key} must have shape ({n},)")
    if np.any(d["train"] & d["valid"]) or np.any(d["conditioning"] & (d["train"] | d["valid"])):
        raise ValueError("Conditioning, training and validation masks must not overlap")
    if "minute_end" in d:
        if d["minute_end"].shape != (n,) or not np.isfinite(d["minute_end"]).all():
            raise ValueError("minute_end must contain one finite minute endpoint per row")
    return d


def likelihood_indices(data: dict[str, np.ndarray], likelihood_minutes: int = 5) -> np.ndarray:
    """Minute endpoints, not averaged temperatures or setpoint-change events.

    ``minute_end`` preserves the clock phase for an explicitly sliced smoke
    dataset. Without it, the first row is assumed to end at minute one.
    Production input has all July rows and both definitions are identical.
    """
    minutes = int(likelihood_minutes)
    if minutes != likelihood_minutes or minutes != 5:
        raise ValueError("This reduction protocol retains the Case 5 five-minute likelihood")
    n = len(data["y"])
    endpoint = np.asarray(data.get("minute_end", np.arange(n) + 1))
    if not np.all(endpoint == np.rint(endpoint)):
        raise ValueError("minute_end must contain integer minute endpoints")
    return np.flatnonzero(np.asarray(data["train"], dtype=bool) & (endpoint % minutes == 0)).astype(np.int64)


def noise_at_interval(sigma_hour: float, rho_hour: float, minutes: float = 5.0) -> tuple[float, float]:
    """Stationary positive-correlation AR(1) mapping from an hourly reference."""
    if not (np.isfinite(sigma_hour) and sigma_hour > 0 and np.isfinite(rho_hour) and 0 < rho_hour < 1):
        raise ValueError("Hourly innovation SD must be positive and 0 < rho_hour < 1")
    if not np.isfinite(minutes) or minutes <= 0:
        raise ValueError("Noise interval must be positive")
    logrho = np.log(rho_hour)
    phi = float(np.exp(logrho * minutes / 60.0))
    variance_ratio = np.expm1(2.0 * logrho * minutes / 60.0) / np.expm1(2.0 * logrho)
    return phi, float(sigma_hour * np.sqrt(variance_ratio))


@dataclass
class PreparedModel:
    """Picklable dense arrays; absent boundary ports have zero conductance."""
    r_boundary: np.ndarray
    r_air: np.ndarray
    capacity: np.ndarray
    window_R: np.ndarray
    window_boundary: np.ndarray
    alpha: np.ndarray
    air_capacity: float
    massive_ids: tuple[str, ...]
    window_ids: tuple[str, ...]


def prepare_model(model: dict[str, Any]) -> PreparedModel:
    massive = [b for b in model["blocks"] if b["kind"] == "massive"]
    windows = [b for b in model["blocks"] if b["kind"] in ("window", "massless")]
    if len(massive) + len(windows) != len(model["blocks"]):
        raise ValueError("Block kind must be massive, window or massless")
    ids = [b["id"] for b in model["blocks"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Block IDs must be unique")
    rb = np.full((len(massive), 4), np.inf, dtype=np.float64)
    for i, b in enumerate(massive):
        if len(b["r_boundary"]) != 4:
            raise ValueError("Every massive block must have four boundary slots")
        active = 0
        for j, value in enumerate(b["r_boundary"]):
            if value is not None:
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("Active boundary resistance must be finite and positive")
                rb[i, j] = value
                active += 1
        if active == 0:
            raise ValueError("A massive block must retain at least one boundary port")
    ra = np.array([b["r_air"] for b in massive], dtype=np.float64)
    c = np.array([b["C"] for b in massive], dtype=np.float64)
    # A grouped massless block can carry several distinct boundary-to-air
    # branches. Flatten branches without replacing their boundary temperatures.
    direct_values, direct_boundaries, direct_ids = [], [], []
    for block in windows:
        if block["kind"] == "window":
            boundary = int(block["boundary"])
            if boundary not in range(4):
                raise ValueError("Window boundary index must lie in [0, 3]")
            ports = [None] * 4
            ports[boundary] = block["R"]
        else:
            ports = block["r_direct"]
            if len(ports) != 4 or all(r is None for r in ports):
                raise ValueError("Massless block needs four slots and an active direct port")
        for boundary, resistance in enumerate(ports):
            if resistance is not None:
                direct_values.append(resistance)
                direct_boundaries.append(boundary)
                direct_ids.append(block["id"])
    wr = np.asarray(direct_values, dtype=np.float64)
    wb = np.asarray(direct_boundaries, dtype=np.int64)
    ca = float(model["air"]["C"])
    if any(np.any(~np.isfinite(a)) or np.any(a <= 0) for a in (ra, c, wr)) or not np.isfinite(ca) or ca <= 0:
        raise ValueError("Every resistance and state capacitance must be finite and positive")
    if np.any((wb < 0) | (wb > 3)):
        raise ValueError("Window boundary index must lie in [0, 3]")
    alpha = np.array([b["alpha"] for b in massive] + [model["air"]["alpha"]], dtype=np.float64)
    if alpha.shape != (len(massive) + 1, 4) or not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("Heat allocations must be four nonnegative finite fractions per state")
    if not np.allclose(alpha.sum(axis=0), 1.0, rtol=0, atol=1e-9):
        raise ValueError("Each heat source's allocations must sum to one over massive states and air")
    for b in windows:
        if float(b.get("C", 0.0)) != 0.0 or np.any(np.asarray(b.get("alpha", [0.0] * 4)) != 0):
            raise ValueError("Massless blocks have no capacitance or separate heat allocation")
    return PreparedModel(rb, ra, c, wr, wb, np.ascontiguousarray(alpha), ca,
                         tuple(b["id"] for b in massive), tuple(direct_ids))


@njit(cache=True, nogil=True)
def _coefficients(rb, ra, capacity, window_R, window_boundary, alpha, air_capacity):
    count = len(ra)
    gb = 1.0 / rb
    gi = 1.0 / ra
    cd = capacity / 30.0
    den = cd + gb.sum(axis=1) + gi
    a = cd / den
    coupling = gi / den
    fb = gb / den.reshape((count, 1))
    fq = alpha[:count] / den.reshape((count, 1))
    direct = np.zeros(4)
    for j in range(len(window_R)):
        direct[window_boundary[j]] += 1.0 / window_R[j]
    fboundary = direct.copy()
    fsource = alpha[count].copy()
    for i in range(count):
        for k in range(4):
            fboundary[k] += gi[i] * fb[i, k]
            fsource[k] += gi[i] * fq[i, k]
    air = air_capacity / 30.0
    dz = air + direct.sum()
    for i in range(count):
        # gi*(1-gi/den) would lose precision when a small positive R_air
        # makes gi dominate; the equivalent expression remains nonnegative.
        dz += gi[i] * (cd[i] + gb[i].sum()) / den[i]
    return a, coupling, gi * a, fb, fq, fboundary, fsource, air, dz


@njit(cache=True, nogil=True)
def _simulate_prepared(rb, ra, capacity, window_R, window_boundary, alpha, air_capacity,
                       boundary, hvac, supply, gains, infiltration, initial, save_states):
    a, c, b, fb, fq, fboundary, fsource, air, dz = _coefficients(
        rb, ra, capacity, window_R, window_boundary, alpha, air_capacity)
    count = len(ra)
    n = len(hvac)
    x = np.full(count, initial)
    z = initial
    prediction = np.empty(n)
    states = np.empty((n, count + 1)) if save_states else np.empty((0, 0))
    forcing = np.empty(count)
    for t in range(n):
        F = hvac[t] * supply[t] + infiltration[t] * boundary[t, 1]
        for k in range(4):
            F += fboundary[k] * boundary[t, k] + fsource[k] * gains[t, k]
        for i in range(count):
            forcing[i] = 0.0
            for k in range(4):
                forcing[i] += fb[i, k] * boundary[t, k] + fq[i, k] * gains[t, k]
        inverse = 1.0 / (dz + hvac[t] + infiltration[t])
        for sub in range(2):
            rhs = air * z + F
            for i in range(count):
                rhs += b[i] * x[i]
            z = rhs * inverse
            for i in range(count):
                x[i] = a[i] * x[i] + c[i] * z + forcing[i]
        prediction[t] = z
        if save_states:
            for i in range(count):
                states[t, i] = x[i]
            states[t, count] = z
    return prediction, states


@njit(cache=True, nogil=True)
def _train_logp(log_values, rb0, ra0, c0, wr0, wb, alpha0, air_capacity,
                kinds, rows, columns, prior_logmean, alpha_budget, prior_a, prior_b,
                boundary, hvac, supply, gains,
                infiltration, initial, y, indices, contiguous, phi, sigma5):
    rb = rb0.copy()
    ra = ra0.copy()
    capacity = c0.copy()
    wr = wr0.copy()
    alpha = alpha0.copy()
    prior = 0.0
    for j in range(len(log_values)):
        value = log_values[j]
        if not np.isfinite(value):
            return -np.inf
        if kinds[j] == 4:
            # Beta(p | a,b), p=sigmoid(z), includes dp/dz. Constants cancel
            # within a target: log target = a*log(p) + b*log(1-p).
            # Calculate both tails directly to avoid subtracting p from one.
            log_p = -np.logaddexp(0.0, -value)
            log_q = -np.logaddexp(0.0, value)
            alpha[rows[j], columns[j]] = alpha_budget[j] * np.exp(log_p)
            alpha[len(ra), columns[j]] = alpha_budget[j] * np.exp(log_q)
            prior += prior_a[j] * log_p + prior_b[j] * log_q
            continue
        if abs(value) > 80:
            return -np.inf
        physical = np.exp(value)
        if kinds[j] == 0:
            rb[rows[j], columns[j]] = physical
        elif kinds[j] == 1:
            ra[rows[j]] = physical
        elif kinds[j] == 2:
            capacity[rows[j]] = physical
        elif kinds[j] == 3:
            wr[rows[j]] = physical
        else:
            return -np.inf
        difference = value - prior_logmean[j]
        prior -= 0.5 * difference * difference / LOG_VARIANCE
    a, c, b, fb, fq, fboundary, fsource, air, dz = _coefficients(
        rb, ra, capacity, wr, wb, alpha, air_capacity)
    count = len(ra)
    x = np.full(count, initial)
    z = initial
    forcing = np.empty(count)
    pointer = 0
    previous_error = 0.0
    quadratic = 0.0
    blocks = 0
    # No full trajectory allocation, and no validation values enter this loop.
    for t in range(indices[-1] + 1):
        F = hvac[t] * supply[t] + infiltration[t] * boundary[t, 1]
        for k in range(4):
            F += fboundary[k] * boundary[t, k] + fsource[k] * gains[t, k]
        for i in range(count):
            forcing[i] = 0.0
            for k in range(4):
                forcing[i] += fb[i, k] * boundary[t, k] + fq[i, k] * gains[t, k]
        denominator = dz + hvac[t] + infiltration[t]
        if not np.isfinite(denominator) or denominator <= 0:
            return -np.inf
        inverse = 1.0 / denominator
        for sub in range(2):
            rhs = air * z + F
            for i in range(count):
                rhs += b[i] * x[i]
            z = rhs * inverse
            for i in range(count):
                x[i] = a[i] * x[i] + c[i] * z + forcing[i]
        if t == indices[pointer]:
            if not np.isfinite(z):
                return -np.inf
            error = z - y[t]
            if contiguous[pointer]:
                innovation = error - phi * previous_error
                quadratic += innovation * innovation
            else:
                quadratic += (1.0 - phi * phi) * error * error
                blocks += 1
            previous_error = error
            pointer += 1
    value = prior - 0.5 * quadratic / (sigma5 * sigma5) - len(indices) * np.log(sigma5)
    value += 0.5 * blocks * np.log1p(-phi * phi)
    return value if np.isfinite(value) else -np.inf


def _simulation_arguments(prepared: PreparedModel, data: dict[str, np.ndarray]):
    p, d = prepared, data
    return (p.r_boundary, p.r_air, p.capacity, p.window_R, p.window_boundary,
            p.alpha, p.air_capacity, d["boundaries"], d["g_hvac"], d["tsupply"],
            d["gains"], d["ginf"], float(d["y"][0]))


def simulate(model: dict[str, Any], data: dict[str, Any], returnstates: bool = False,
             *, return_states: bool | None = None):
    """Return end-of-minute T_air, optionally ``(T_air, all_state_values)``.

    State columns follow massive-block order in the model, then zone air.
    States are continuous across all split boundaries. The optional alias
    ``return_states`` has the same meaning as ``returnstates``.
    """
    if return_states is not None:
        returnstates = bool(return_states)
    prepared, d = prepare_model(model), _validated_data(data)
    prediction, states = _simulate_prepared(*_simulation_arguments(prepared, d), bool(returnstates))
    if not np.isfinite(prediction).all() or (returnstates and not np.isfinite(states).all()):
        raise FloatingPointError("Nonfinite thermal trajectory")
    return (prediction, states) if returnstates else prediction


@dataclass
class ReducedTarget:
    """Picklable conditional log-posterior, with a Numba nogil hot path."""
    model: dict[str, Any]
    free_specs: list[dict[str, Any]]
    data: dict[str, Any]
    likelihood_minutes: int = 5

    def __post_init__(self):
        self.model = deepcopy(self.model)
        self.free_specs = deepcopy(self.free_specs)
        self.data = _validated_data(self.data)
        self.prepared = prepare_model(self.model)
        self.ndim = len(self.free_specs)
        self.names = [s["name"] for s in self.free_specs]
        if len(set(self.names)) != self.ndim:
            raise ValueError("Free parameter names must be unique")
        # Retain the legacy attribute name for sampler compatibility. Its
        # entries are now mixed log(R/C) and logit(alpha / budget) coordinates.
        self.initial_log_values = np.empty(self.ndim, dtype=np.float64)
        self.prior_logmean = np.zeros(self.ndim, dtype=np.float64)
        self.alpha_budget = np.zeros(self.ndim, dtype=np.float64)
        self.prior_a = np.zeros(self.ndim, dtype=np.float64)
        self.prior_b = np.zeros(self.ndim, dtype=np.float64)
        self.coordinate_labels = []
        self.diagnostic_names = list(self.names)
        self.indices = likelihood_indices(self.data, self.likelihood_minutes)
        if len(self.indices) == 0:
            raise ValueError("No training likelihood endpoints; do not relabel conditioning data as training")
        self.contiguous = np.r_[False, np.diff(self.indices) == self.likelihood_minutes]
        # Explicit time_seconds can detect physical gaps in a sliced input.
        if "time_seconds" in self.data:
            times = np.asarray(self.data["time_seconds"])
            if times.shape != self.data["y"].shape or np.any(np.diff(times) != 60):
                raise ValueError("Physical input rows must be contiguous one-minute intervals")
        self.phi5, self.sigma5 = noise_at_interval(float(self.model["noise"]["sigma_hour"]),
                                                  float(self.model["noise"]["rho_hour"]), self.likelihood_minutes)
        massive = {name: i for i, name in enumerate(self.prepared.massive_ids)}
        direct = {(name, int(boundary)): i for i, (name, boundary) in enumerate(
            zip(self.prepared.window_ids, self.prepared.window_boundary))}
        original_windows = {b["id"]: int(b["boundary"]) for b in self.model["blocks"]
                            if b["kind"] == "window"}
        kinds, rows, columns, destinations, alpha_sources = [], [], [], set(), set()
        for j, spec in enumerate(self.free_specs):
            block, field = spec["block_id"], spec["field"]
            index = int(spec.get("index", -1))
            destination = (block, field, index)
            if destination in destinations:
                raise ValueError("A physical parameter cannot appear twice in free_specs")
            destinations.add(destination)
            if field == "r_boundary" and block in massive and 0 <= index < 4:
                row = massive[block]
                if not np.isfinite(self.prepared.r_boundary[row, index]):
                    raise ValueError("A free boundary resistance must refer to an existing active port")
                kind = 0
            elif field == "r_air" and block in massive:
                kind, row = 1, massive[block]
            elif field == "C" and block in massive:
                kind, row = 2, massive[block]
            elif field == "R" and block in original_windows:
                kind, row = 3, direct[(block, original_windows[block])]
            elif field == "r_direct" and (block, index) in direct:
                kind, row = 3, direct[(block, index)]
            elif field == "alpha_fraction" and block in massive and 0 <= index < 4:
                if index in alpha_sources:
                    raise ValueError("Only one merged-to-air split per source may be sampled")
                alpha_sources.add(index)
                kind, row = 4, massive[block]
            else:
                raise ValueError(f"Unsupported free parameter: {spec}")
            initial = float(spec["initial"])
            if kind == 4:
                if spec.get("transform", "logit") != "logit":
                    raise ValueError("Alpha fractions require logit coordinates")
                budget = float(spec["budget"])
                a, b = float(spec["prior_a"]), float(spec["prior_b"])
                if not (np.isfinite(budget) and 0 < budget <= 1.0 + 1e-9
                        and np.isfinite(initial) and 0 < initial < budget
                        and np.isfinite(a) and a > 0 and np.isfinite(b) and b > 0):
                    raise ValueError("Alpha split requires a positive budget, an interior initial value and positive Beta shapes")
                available = self.prepared.alpha[row, index] + self.prepared.alpha[-1, index]
                if not np.isclose(budget, available, rtol=0, atol=1e-9):
                    raise ValueError("Alpha budget must equal the fixed merged-plus-air source allocation")
                self.initial_log_values[j] = np.log(initial) - np.log(budget - initial)
                self.alpha_budget[j], self.prior_a[j], self.prior_b[j] = budget, a, b
                self.coordinate_labels.append("logit")
            else:
                if spec.get("transform", "log") != "log":
                    raise ValueError("Positive R/C parameters require log coordinates")
                mean = float(spec["prior_mean"])
                if not (np.isfinite(initial) and initial > 0 and np.isfinite(mean) and mean > 0):
                    raise ValueError("R/C initial values and aggregate IDF prior means must be positive and finite")
                self.initial_log_values[j] = np.log(initial)
                self.prior_logmean[j] = np.log(mean) - 0.5 * LOG_VARIANCE
                self.coordinate_labels.append("log")
            kinds.append(kind)
            rows.append(row)
            columns.append(index)
        self.kinds = np.asarray(kinds, dtype=np.int64)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.columns = np.asarray(columns, dtype=np.int64)
        self.coordinate_labels = tuple(self.coordinate_labels)
        digest = hashlib.sha256()
        digest.update(json.dumps({"model": self.model, "free_specs": self.free_specs,
                                  "likelihood_minutes": self.likelihood_minutes,
                                  "target_schema": "conditional-allpairs-RCA-log-logit-be30s-v2"},
                                 sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        for key in sorted(self.data):
            value = np.ascontiguousarray(self.data[key])
            digest.update(key.encode())
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        self.fingerprint = digest.hexdigest()

    def logp(self, log_values) -> float:
        values = np.asarray(log_values, dtype=np.float64)
        if values.shape != (self.ndim,) or not np.isfinite(values).all():
            return -np.inf
        p, d = self.prepared, self.data
        return float(_train_logp(values, p.r_boundary, p.r_air, p.capacity, p.window_R,
                                p.window_boundary, p.alpha, p.air_capacity, self.kinds,
                                self.rows, self.columns, self.prior_logmean, self.alpha_budget,
                                self.prior_a, self.prior_b, d["boundaries"],
                                d["g_hvac"], d["tsupply"], d["gains"], d["ginf"],
                                float(d["y"][0]), d["y"], self.indices, self.contiguous,
                                self.phi5, self.sigma5))

    def physical_values(self, coordinates) -> np.ndarray:
        """Decode (..., ndim) unconstrained draws into actual R, C and alpha.

        Alpha outputs are the merged fractions, not logits or fractions of the
        merged-plus-air budget. The air allocation is a derived remainder and
        does not introduce another independent sampling coordinate.
        """
        values = np.asarray(coordinates, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != self.ndim:
            raise ValueError("Last coordinate dimension must match the free parameter count")
        result = np.empty_like(values)
        positive = self.kinds != 4
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result[..., positive] = np.exp(values[..., positive])
            for j in np.flatnonzero(~positive):
                result[..., j] = self.alpha_budget[j] * np.exp(-np.logaddexp(0.0, -values[..., j]))
        return result

    def model_at(self, log_values) -> dict[str, Any]:
        values = np.asarray(log_values, dtype=np.float64)
        if (values.shape != (self.ndim,) or not np.isfinite(values).all()
                or np.any(np.abs(values[self.kinds != 4]) > 80)):
            raise ValueError("Invalid reduced-model unconstrained parameter vector")
        physical = self.physical_values(values)
        result = deepcopy(self.model)
        blocks = {b["id"]: b for b in result["blocks"]}
        for j, (spec, value) in enumerate(zip(self.free_specs, physical)):
            block = blocks[spec["block_id"]]
            field = spec["field"]
            if field in ("r_boundary", "r_direct"):
                block[field][int(spec["index"])] = float(value)
            elif field == "alpha_fraction":
                source = int(spec["index"])
                block["alpha"][source] = float(value)
                result["air"]["alpha"][source] = float(
                    self.alpha_budget[j] * np.exp(-np.logaddexp(0.0, values[j])))
            else:
                block[field] = float(value)
        return result

    def predict(self, log_values) -> np.ndarray:
        return simulate(self.model_at(log_values), self.data)
