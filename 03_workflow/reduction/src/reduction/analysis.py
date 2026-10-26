"""Prediction metrics and exploratory subgroup discovery.

Binary discovery uses beam search over short conjunctions, ranked by
unweighted WRAcc. The WRAcc definition follows Lavrac et al. (2004), §2.2:
https://www.jmlr.org/papers/volume5/lavrac04a/lavrac04a.pdf
The search differs from CN2-SD's weighted covering procedure.

Selectors use an explicit allowlist of prefit pair descriptors. Results
describe the evaluated HIBO July reduction path; statistical significance
and generalization across buildings, periods, or paths are not established."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PREFIT_FEATURES = (
    "pair_d_R", "pair_d_C", "R_eff_left", "R_eff_right", "C_left", "C_right",
    "window_pair", "operator", "boundary_count_before", "boundary_count_after",
    "parent_n_R", "parent_n_C", "child_n_R", "child_n_C",
)
_CATEGORICAL = {"window_pair", "operator"}
_RESOLVED = {"pass", "fail"}
_EXCLUDED = {"unresolved", "stopped"}
_SCOPE = (
    "Exploratory descriptions of this HIBO Lab B July experiment and the evaluated "
    "local-greedy reduction path only. Trials share the same building, forcing, "
    "validation period, and sequential parent models; they are not independent "
    "building or time replications. July 23-31 is repeatedly used for model "
    "selection and is not an untouched test set. No statistical-significance, "
    "causal, global-optimum, or out-of-sample generalization claim is made. "
    "A pass requires both convergence and the configured accuracy gate; "
    "unresolved/stopped trials are not physical failures."
)


def _first(data: Mapping, keys: Iterable[str], *, required: bool = True) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    if required:
        raise KeyError("Required data key missing: one of " + ", ".join(keys))
    return None


def _vector(value: Any, n: int, name: str, dtype=float) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"{name} must be a one-dimensional vector of length {n}")
    return arr


def _error_metrics(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> dict:
    n = int(np.sum(mask))
    if not n:
        return {"n": 0, "rmse": None, "mae": None, "bias": None,
                "max_abs_error": None, "observed_min": None, "observed_max": None,
                "observed_std": None, "predicted_min": None, "predicted_max": None,
                "predicted_std": None, "p95_abs_error": None}
    y, p = obs[mask], pred[mask]
    e = p - y
    return {
        "n": n, "rmse": float(np.sqrt(np.mean(e * e))),
        "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
        "max_abs_error": float(np.max(np.abs(e))),
        "p95_abs_error": float(np.quantile(np.abs(e), 0.95)),
        "observed_min": float(np.min(y)), "observed_max": float(np.max(y)),
        "observed_std": float(np.std(y)), "predicted_min": float(np.min(p)),
        "predicted_max": float(np.max(p)), "predicted_std": float(np.std(p)),
    }


def prediction_metrics(pred: Any, data: Mapping[str, Any]) -> dict:
    """Evaluate minute-end predictions, preserving train/selection separation.

    Input contract matches inputs/data.npz: y, train, valid, dt_seconds; optional
    time_seconds, heating_setpoint, cooling_setpoint. train_mask and
    validation_mask are accepted aliases. Five-minute metrics use endpoints,
    never five-minute means or a re-run with five-minute integration steps.

    Variety diagnostics are descriptions, not additional tuned acceptance gates.
    Observed-temperature bins form a disjoint partition including out-of-range
    values. Setpoint phases are 0-30 min, 30-120 min, and >=120 min after an
    observed setpoint change; time before the first observed change is unknown.
    """
    obs = np.asarray(_first(data, ("y", "temperature_observed", "observed")), dtype=float)
    if obs.ndim != 1 or not obs.size:
        raise ValueError("Observed temperatures must be a nonempty one-dimensional vector")
    n = len(obs)
    p = _vector(pred, n, "prediction")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(obs)):
        raise ValueError("Nonfinite predictions/observations cannot be silently omitted from metrics")
    train = _vector(_first(data, ("train", "train_mask")), n, "train", bool)
    valid = _vector(_first(data, ("valid", "validation_mask", "validation")), n, "valid", bool)
    if np.any(train & valid):
        raise ValueError("Training and selection-validation masks overlap")
    conditioning_raw = _first(data, ("conditioning", "conditioning_mask"), required=False)
    conditioning = (~(train | valid) if conditioning_raw is None
                    else _vector(conditioning_raw, n, "conditioning", bool))
    if np.any(conditioning & (train | valid)):
        raise ValueError("Conditioning samples must not enter train/validation metrics")
    dt = float(data.get("dt_seconds", 60.0))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt_seconds must be positive")
    raw_time = data.get("time_seconds")
    time = ((np.arange(n, dtype=float) + 1) * dt if raw_time is None
            else _vector(raw_time, n, "time_seconds"))
    if not np.all(np.isfinite(time)) or (n > 1 and np.any(np.diff(time) <= 0)):
        raise ValueError("time_seconds must be finite and strictly increasing")
    if n > 1 and not np.allclose(np.diff(time), dt, atol=1e-6, rtol=1e-9):
        raise ValueError("Prediction metrics require regular samples matching dt_seconds")
    endpoint = np.isclose(np.mod(time, 300.0), 0.0, atol=1e-6, rtol=0)
    endpoint |= np.isclose(np.mod(time, 300.0), 300.0, atol=1e-6, rtol=0)
    out = {
        "sampling_interval_seconds": dt,
        "prediction_type": "one coherent parameter vector unless caller states otherwise",
        "conditioning_excluded": True,
        "n_conditioning": int(conditioning.sum()),
        "n_unassigned": int((~(train | valid | conditioning)).sum()),
        "validation_role": "repeated model-selection validation, not untouched test",
        "train": _error_metrics(p, obs, train),
        "validation": _error_metrics(p, obs, valid),
        "five_minute": {
            "definition": "Every 300-second endpoint from the same minute-resolution simulation",
            "train": _error_metrics(p, obs, train & endpoint),
            "validation": _error_metrics(p, obs, valid & endpoint),
        },
        "variety": {},
    }
    heat_raw = _first(data, ("heating_setpoint", "heat_setpoint"), required=False)
    cool_raw = _first(data, ("cooling_setpoint", "cool_setpoint"), required=False)
    setpoints = None
    if heat_raw is not None and cool_raw is not None:
        heat = _vector(heat_raw, n, "heating_setpoint")
        cool = _vector(cool_raw, n, "cooling_setpoint")
        if not np.all(np.isfinite(heat)) or not np.all(np.isfinite(cool)):
            raise ValueError("Setpoint records contain nonfinite values")
        if np.any(heat > cool):
            raise ValueError("Heating setpoint exceeds cooling setpoint")
        setpoints = np.column_stack([heat, cool])
        changes = np.zeros(n, dtype=bool)
        changes[1:] = np.any(np.abs(np.diff(setpoints, axis=0)) > 1e-8, axis=1)
        last_change = np.maximum.accumulate(np.where(changes, time, -np.inf))
        elapsed_min = (time - last_change) / 60.0
        previous = np.roll(0.5 * (heat + cool), 1)
        direction = np.zeros(n, dtype=np.int8)
        change_idx = np.flatnonzero(changes)
        last_idx = np.maximum.accumulate(np.where(changes, np.arange(n), -1))
        known = last_idx >= 0
        if len(change_idx):
            direction[known] = np.sign(
                0.5 * (heat[last_idx[known]] + cool[last_idx[known]])
                - previous[last_idx[known]]).astype(np.int8)
    temperature_masks = (
        ("<20", obs < 20), ("20-22", (obs >= 20) & (obs < 22)),
        ("22-24", (obs >= 22) & (obs < 24)),
        ("24-26", (obs >= 24) & (obs < 26)),
        ("26-28", (obs >= 26) & (obs <= 28)), (">28", obs > 28),
    )
    for name, partition in (("train", train), ("validation", valid)):
        v = {
            "temperature_bins": [
                {"bin_C": label, **_error_metrics(p, obs, partition & mask)}
                for label, mask in temperature_masks
            ],
            "temperature_bin_definition": "E+ observed temperature; left inclusive/right exclusive, 28 included in 26-28",
            "setpoint_available": setpoints is not None,
        }
        if setpoints is None:
            v["setpoint_note"] = "Both heating and cooling setpoint arrays are required; no schedule was inferred"
        else:
            v["setpoint_levels"] = [
                {"heating_setpoint_C": float(pair[0]), "cooling_setpoint_C": float(pair[1]),
                 **_error_metrics(p, obs, partition & np.all(np.isclose(setpoints, pair, rtol=0, atol=1e-8), axis=1))}
                for pair in np.unique(setpoints[partition], axis=0)
            ]
            phase_masks = (
                ("0-30 min after change", known & (elapsed_min < 30)),
                ("30-120 min after change", known & (elapsed_min >= 30) & (elapsed_min < 120)),
                (">=120 min after change", known & (elapsed_min >= 120)),
                ("Before first observed change", ~known),
            )
            v["transition_phases"] = [
                {"phase": label, **_error_metrics(p, obs, partition & mask)}
                for label, mask in phase_masks
            ]
            v["transition_direction"] = [
                {"direction": label, **_error_metrics(p, obs, partition & mask)}
                for label, mask in (
                    ("upward setpoint midpoint", known & (direction > 0)),
                    ("downward setpoint midpoint", known & (direction < 0)),
                    ("unchanged midpoint / changed deadband", known & (direction == 0)),
                    ("unknown before first change", ~known),
                )
            ]
            v["observed_setpoint_changes"] = int(np.sum(changes & partition))
            v["phase_definition"] = (
                "Elapsed time from recorded minute-end heating/cooling setpoint changes; "
                "up/down uses their midpoint. Changes before this dataset are unknown. "
                "These fixed descriptive windows are not physical settling-time claims."
            )
        out["variety"][name] = v
    return out


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def _normalize_rows(rows: Iterable[Mapping]) -> list[dict]:
    result, ids = [], set()
    for row in rows:
        trial_id = str(row["trial_id"])
        if trial_id in ids:
            raise ValueError(f"Duplicate trial_id would inflate rule support: {trial_id}")
        ids.add(trial_id)
        status = str(row.get("status", "")).lower()
        if status not in _RESOLVED | _EXCLUDED:
            raise ValueError(f"Unknown rule outcome {status!r} for {trial_id}")
        source = row.get("features", {})
        features = {}
        for key in PREFIT_FEATURES:
            value = source.get(key)
            if key in _CATEGORICAL:
                features[key] = (None if value is None else bool(value)) if key == "window_pair" else (None if value is None else str(value))
            else:
                features[key] = float(value) if _finite(value) else None
        result.append({"trial_id": trial_id, "status": status,
                       "depth": row.get("depth"), "features": features})
    return result


def _condition_text(cond: Mapping) -> str:
    value = cond["value"]
    shown = str(value) if isinstance(value, (bool, str)) else f"{value:.8g}"
    return f"{cond['feature']} {cond['operator']} {shown}"


def _match(feature: Mapping, cond: Mapping) -> bool:
    value = feature.get(cond["feature"])
    if value is None:
        return False
    op, ref = cond["operator"], cond["value"]
    if op == "==":
        return value == ref
    return value <= ref if op == "<=" else value > ref


def _selectors(rows: list[dict], features: list[str], max_thresholds: int) -> list[dict]:
    """Generate thresholds solely from resolved prefit descriptors, not labels."""
    selectors = []
    resolved = [r for r in rows if r["status"] in _RESOLVED]
    for feature in features:
        if feature in _CATEGORICAL:
            values = sorted({r["features"][feature] for r in resolved
                             if r["features"].get(feature) is not None}, key=str)
            if len(values) < 2:
                continue
            selectors.extend({"feature": feature, "operator": "==", "value": x} for x in values)
        else:
            values = np.unique([r["features"][feature] for r in resolved
                                if r["features"].get(feature) is not None])
            if len(values) < 2:
                continue
            cuts = (values[:-1] / 2.0 + values[1:] / 2.0)
            if len(cuts) > max_thresholds:
                positions = np.unique(np.rint(np.linspace(0, len(cuts) - 1, max_thresholds)).astype(int))
                cuts = cuts[positions]
            for cut in cuts:
                selectors.extend({"feature": feature, "operator": op, "value": float(cut)}
                                 for op in ("<=", ">"))
    return selectors


def _compatible(conditions: list[dict]) -> bool:
    """Allow useful numeric intervals, reject redundant/contradictory repeats."""
    by_feature: dict[str, list[dict]] = {}
    for cond in conditions:
        by_feature.setdefault(cond["feature"], []).append(cond)
    for group in by_feature.values():
        if len(group) == 1:
            continue
        if len(group) != 2 or {c["operator"] for c in group} != {"<=", ">"}:
            return False
        lower = next(c["value"] for c in group if c["operator"] == ">")
        upper = next(c["value"] for c in group if c["operator"] == "<=")
        if lower >= upper:
            return False
    return True


def _rule_record(indices: tuple[int, ...], selectors: list[dict], rows: list[dict],
                 cover: np.ndarray, target: str, n_resolved: int, baseline: float) -> dict:
    matching = [r for r, flag in zip(rows, cover) if flag]
    resolved = [r for r in matching if r["status"] in _RESOLVED]
    supporting = [r["trial_id"] for r in resolved if r["status"] == target]
    exceptions = [r["trial_id"] for r in resolved if r["status"] != target]
    n = len(resolved)
    rate = len(supporting) / n
    conditions = [selectors[i] for i in indices]
    excluded = [r for r in matching if r["status"] in _EXCLUDED]
    return {
        "target": target, "conditions": conditions,
        "condition_text": " AND ".join(_condition_text(c) for c in conditions),
        "n_conditions": len(conditions), "resolved_coverage": n,
        "target_count": len(supporting), "exception_count": len(exceptions),
        "target_rate": rate, "baseline_rate": baseline,
        "coverage_fraction": n / n_resolved,
        "wracc": (n / n_resolved) * (rate - baseline),
        "covered_resolved_trial_ids": [r["trial_id"] for r in resolved],
        "supporting_trial_ids": supporting, "exception_trial_ids": exceptions,
        "excluded_count": len(excluded),
        "excluded_trial_ids": [r["trial_id"] for r in excluded],
        "excluded_by_status": dict(Counter(r["status"] for r in excluded)),
        "exception_statement": (
            "No exceptions among the covered resolved trials; not a universal or proven rule."
            if not exceptions else "Exceptions are the covered resolved trials with the opposite outcome."
        ),
    }


def _rank(rule: Mapping) -> tuple:
    return (-rule["wracc"], -rule["resolved_coverage"], rule["n_conditions"], rule["condition_text"])


def discover_rules(rows: Iterable[Mapping], config: Mapping[str, Any] | None,
                   output_dir: str | Path) -> dict:
    """Discover overlapping short conjunctions for PASS and FAIL separately.

    The output includes every rule's supporting IDs, exceptions, and matching
    unresolved/stopped trials. The latter never enter WRAcc denominators.
    Rule support means covered *resolved* trials. Repeated coverage is reported
    explicitly; rows are never treated as disjoint just because rules differ.
    """
    cfg = dict(config or {})
    if "rules" in cfg:
        cfg = dict(cfg["rules"])
    unknown_config = set(cfg) - {
        "beam_width", "max_depth", "min_support", "min_resolved",
        "max_rules_per_outcome", "max_thresholds_per_feature", "features",
    }
    if unknown_config:
        raise ValueError("Unknown rule configuration keys: " + ", ".join(sorted(unknown_config)))
    beam_width = int(cfg.get("beam_width", 20))
    max_depth = int(cfg.get("max_depth", 2))
    min_support = int(cfg.get("min_support", 5))
    min_resolved = int(cfg.get("min_resolved", 20))
    max_rules = int(cfg.get("max_rules_per_outcome", 10))
    max_thresholds = int(cfg.get("max_thresholds_per_feature", 12))
    if max_depth not in (1, 2, 3):
        raise ValueError("max_depth must be 1, 2, or 3 for readable exploratory rules")
    if min(beam_width, min_support, min_resolved, max_rules, max_thresholds) < 1:
        raise ValueError("Beam width, support, resolved count, rule count, and threshold count must be positive")
    features = list(cfg.get("features", PREFIT_FEATURES))
    unknown = set(features) - set(PREFIT_FEATURES)
    if unknown:
        raise ValueError("Rule-feature leakage/unknown features rejected: " + ", ".join(sorted(unknown)))
    if not features or len(set(features)) != len(features):
        raise ValueError("At least one distinct prefit rule feature is required")
    records = _normalize_rows(rows)
    counts = Counter(r["status"] for r in records)
    n_resolved = counts["pass"] + counts["fail"]
    report = {
        "schema": "hibo_exploratory_rules_v1", "status": "complete", "scope": _SCOPE,
        "algorithm": "Beam search of conjunctions; unweighted WRAcc; not the full CN2-SD covering algorithm",
        "source": {"reference": "Lavrac et al. (2004), Subgroup Discovery with CN2-SD, JMLR 5:153-188, section 2.2",
                   "url": "https://www.jmlr.org/papers/volume5/lavrac04a/lavrac04a.pdf"},
        "wracc_formula": "n_covered_resolved / n_resolved * (target_fraction_in_covered_resolved - global_target_fraction)",
        "counts": {"all_trials": len(records), "resolved": n_resolved,
                   "pass": counts["pass"], "fail": counts["fail"],
                   "unresolved": counts["unresolved"], "stopped": counts["stopped"]},
        "baseline_rates": {key: counts[key] / n_resolved if n_resolved else None for key in ("pass", "fail")},
        "excluded_trial_ids": [r["trial_id"] for r in records if r["status"] in _EXCLUDED],
        "prefit_features": features, "no_postfit_predictors": True,
        "config": {"beam_width": beam_width, "max_depth": max_depth, "min_support": min_support,
                   "min_resolved": min_resolved, "max_rules_per_outcome": max_rules,
                   "max_thresholds_per_feature": max_thresholds},
        "rules": [], "overlap": {},
        "support_definition": "Number of covered resolved pass/fail trials, not all matching trials",
        "missing_features": {f: sum(r["features"].get(f) is None for r in records) for f in features},
    }
    if n_resolved < min_resolved:
        report["status"] = "insufficient_resolved_trials"
        report["reason"] = f"{n_resolved} resolved trials; at least {min_resolved} configured. No rules inferred."
    elif not counts["pass"] or not counts["fail"]:
        report["status"] = "no_outcome_contrast"
        report["reason"] = "Both resolved outcomes are required to discover enrichment; no rules inferred."
    else:
        selectors = _selectors(records, features, max_thresholds)
        selector_masks = [np.array([_match(r["features"], s) for r in records], dtype=bool) for s in selectors]
        resolved_mask = np.array([r["status"] in _RESOLVED for r in records])
        report["selector_count"] = len(selectors)
        for target in ("pass", "fail"):
            baseline = counts[target] / n_resolved
            beam = [tuple()]
            candidates: dict[tuple[str, ...], dict] = {}
            for depth in range(1, max_depth + 1):
                generation: dict[tuple[int, ...], tuple[dict, np.ndarray]] = {}
                for prefix in beam:
                    for idx in range(prefix[-1] + 1 if prefix else 0, len(selectors)):
                        indices = prefix + (idx,)
                        if not _compatible([selectors[i] for i in indices]):
                            continue
                        mask = np.logical_and.reduce([selector_masks[i] for i in indices])
                        if int(np.sum(mask & resolved_mask)) < min_support:
                            continue
                        rule = _rule_record(indices, selectors, records, mask, target, n_resolved, baseline)
                        generation[indices] = (rule, mask)
                        # Deduplicate observationally identical coverage but keep exclusions too.
                        signature = tuple(rule["covered_resolved_trial_ids"] + ["|excluded|"] + rule["excluded_trial_ids"])
                        if rule["wracc"] > 1e-15:
                            old = candidates.get(signature)
                            if old is None or _rank(rule) < _rank(old):
                                candidates[signature] = rule
                # Neutral prefixes may yield informative conjunctions; do not require positive WRAcc here.
                beam = [indices for indices, _ in sorted(generation.items(), key=lambda item: _rank(item[1][0]))[:beam_width]]
                if not beam:
                    break
            selected = sorted(candidates.values(), key=_rank)[:max_rules]
            for number, rule in enumerate(selected, 1):
                rule["rule_id"] = f"{target.upper()}_{number:02d}"
            report["rules"].extend(selected)
        if not report["rules"]:
            report["status"] = "no_positive_wracc_rules"
            report["reason"] = "No searched conjunction met support and positive WRAcc; no rule is invented."
    for target in ("pass", "fail"):
        covered = Counter(tid for r in report["rules"] if r["target"] == target for tid in r["covered_resolved_trial_ids"])
        report["overlap"][target] = {
            "unique_covered_resolved_trials": len(covered),
            "sum_rule_coverages": int(sum(covered.values())),
            "trials_in_multiple_rules": {tid: n for tid, n in sorted(covered.items()) if n > 1},
            "note": "Rules may overlap; support counts must not be added as independent observations.",
        }
    _write_rule_report(report, records, Path(output_dir))
    return report


def _write_rule_report(report: dict, records: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "rules.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "rule_inputs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial_id", "depth", "status", *PREFIT_FEATURES])
        writer.writeheader()
        for row in records:
            writer.writerow({"trial_id": row["trial_id"], "depth": row["depth"],
                             "status": row["status"], **row["features"]})
    columns = ["rule_id", "target", "condition_text", "resolved_coverage", "target_count",
               "exception_count", "target_rate", "baseline_rate", "wracc", "excluded_count",
               "supporting_trial_ids", "exception_trial_ids", "excluded_trial_ids"]
    with (out / "rules.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rule in report["rules"]:
            writer.writerow({c: (json.dumps(rule[c]) if isinstance(rule[c], list) else rule[c]) for c in columns})
    count = report["counts"]
    lines = ["# Exploratory reduction rules", "", report["scope"], "",
             f"Status: **{report['status']}**", "",
             f"Trials: {count['all_trials']} total; {count['pass']} pass; {count['fail']} fail; "
             f"{count['unresolved']} unresolved; {count['stopped']} stopped. "
             f"All rates and WRAcc denominators use the {count['resolved']} resolved trials only.", "",
             "Support = covered resolved trials. Supporting IDs match the rule's target; "
             "exception IDs have the opposite resolved outcome. Matching unresolved/stopped trials are listed separately.", "",
             "WRAcc = coverage fraction × (subgroup target rate − overall target rate). "
             "Only prefit pair descriptors are searched. RMSE, posterior diagnostics, fitted child parameters, "
             "and adoption decisions are excluded from predictors.", "",
             f"Configuration: {json.dumps(report['config'], sort_keys=True)}", ""]
    if report.get("reason"):
        lines += [report["reason"], ""]
    for target in ("pass", "fail"):
        lines += [f"## {target.upper()} enrichment", ""]
        target_rules = [r for r in report["rules"] if r["target"] == target]
        if not target_rules:
            lines += ["No rule reported.", ""]
        for rule in target_rules:
            lines += [f"### {rule['rule_id']}: {rule['condition_text']}", "",
                      f"Resolved support: {rule['resolved_coverage']}; target: {rule['target_count']}; "
                      f"exceptions: {rule['exception_count']}; target rate: {rule['target_rate']:.3f}; "
                      f"baseline: {rule['baseline_rate']:.3f}; WRAcc: {rule['wracc']:.6f}.", "",
                      "Supporting trial IDs: " + ", ".join(rule["supporting_trial_ids"]) + ".", "",
                      "Exception trial IDs: " + (", ".join(rule["exception_trial_ids"]) or "None in observed covered resolved trials") + ".", "",
                      rule["exception_statement"], "",
                      "Matching excluded trial IDs: " + (", ".join(rule["excluded_trial_ids"]) or "None") + ".", ""]
        overlap = report["overlap"][target]
        lines += [f"Overlap: {overlap['unique_covered_resolved_trials']} unique covered resolved trials "
                  f"versus {overlap['sum_rule_coverages']} summed rule coverages. " + overlap["note"], ""]
    lines += ["## Method reference", "",
              "WRAcc: Lavrac et al. (2004), *Subgroup Discovery with CN2-SD*, JMLR 5:153–188, section 2.2. "
              "[Primary paper](https://www.jmlr.org/papers/volume5/lavrac04a/lavrac04a.pdf). "
              "This implementation uses a simple beam search, not the paper's full weighted-covering algorithm.", ""]
    (out / "rules.md").write_text("\n".join(lines), encoding="utf-8")
