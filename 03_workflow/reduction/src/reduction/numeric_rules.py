"""Exploratory numeric subgroup discovery from evaluated candidates.

Targets are postfit d_R/d_C and selectors use prefit descriptors. Discovery
is performed after reduction and does not alter likelihoods, PASS labels,
Pareto objectives, or candidate eligibility."""
from __future__ import annotations

import html
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "min_support": 10, "min_stages": 3, "max_coverage": 0.85,
    "max_depth": 2, "max_rules": 30,
}
PREFIT_FEATURES = (
    "pair_d_R", "pair_d_C", "C_min", "C_max", "R_min", "R_max",
    "boundary_count_after", "shared_boundary_count",
)
SCOPE = (
    "Exploratory priority heuristics for evaluated massive--massive PASS trials "
    "on this building, forcing, and sequential reduction path. Trials are not "
    "independent replications. No independent validation, causal claim, "
    "statistical significance, physical admissibility rule, or PASS guarantee "
    "is established. Rules do not prune candidates or feed back into this run."
)
DISTANCE_NOTE = (
    "Parent similarity is partly coupled to the target definition: "
    "d_C^2 = log(C_merged/sqrt(C_left*C_right))^2 + "
    "(log(C_left/C_right)/2)^2; the analogous identity holds for R_eff. "
    "Excess-distance ranks below remove the second term as a sensitivity "
    "description, not as a new MCMC or Pareto objective."
)


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _config(config):
    cfg = dict(DEFAULT_CONFIG)
    unknown = set(config or {}) - set(cfg)
    if unknown:
        raise ValueError("Unknown numeric rule configuration (selector leakage is not allowed): " + ", ".join(sorted(unknown)))
    cfg.update(config or {})
    for key in ("min_support", "min_stages", "max_depth", "max_rules"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["max_depth"] > 2:
        raise ValueError("Numeric rules are restricted to at most two prefit conditions")
    if not isinstance(cfg["max_coverage"], (int, float)) or not 0 < cfg["max_coverage"] < 1:
        raise ValueError("max_coverage must be between zero and one")
    return cfg


def _prepare(rows):
    records, excluded, seen = [], Counter(), set()
    for row in rows:
        tid = str(row["trial_id"])
        if tid in seen:
            raise ValueError(f"Duplicate trial_id would inflate numeric support: {tid}")
        seen.add(tid)
        status = str(row.get("status", "")).lower()
        if status not in {"pass", "fail", "unresolved", "stopped"}:
            raise ValueError(f"Unknown numeric rule outcome {status!r}: {tid}")
        feat = row.get("features", {})
        if feat.get("operator") != "opaque_state_aggregation":
            excluded["other_operator"] += 1
            continue
        if status != "pass":
            excluded["massive_" + status] += 1
            continue
        values = {k: _number(feat.get(k)) for k in ("R_eff_left", "R_eff_right", "C_left", "C_right")}
        if any(x is None or x <= 0 for x in values.values()):
            raise ValueError(f"Invalid positive prefit R/C for massive PASS trial {tid}")
        change = row.get("parameter_changes", {})
        targets = {k: _number(change.get(k)) for k in ("d_R", "d_C")}
        if any(x is None or x < 0 for x in targets.values()):
            raise ValueError(f"Invalid postfit numeric target for massive PASS trial {tid}")
        depth = row.get("depth")
        if depth is None:
            raise ValueError(f"Missing parent stage for {tid}")
        # Hash prevents unrelated parents at the same depth being pooled.
        stage = f"{depth}:{row.get('parent_hash', '')}"
        left = tuple(sorted(str(x) for x in feat.get("left_members", [row.get("left", tid + ':left')])))
        right = tuple(sorted(str(x) for x in feat.get("right_members", [row.get("right", tid + ':right')])))
        pair_key = json.dumps(sorted([left, right]), separators=(",", ":"))
        before, after = (_number(feat.get(k)) for k in ("boundary_count_before", "boundary_count_after"))
        record = dict(trial_id=tid, depth=depth, stage_id=stage, status=status,
                      left=" + ".join(left), right=" + ".join(right), pair_key=pair_key,
                      **values, **targets)
        record.update(
            pair_d_R=abs(math.log(values["R_eff_left"] / values["R_eff_right"])),
            pair_d_C=abs(math.log(values["C_left"] / values["C_right"])),
            C_min=min(values["C_left"], values["C_right"]), C_max=max(values["C_left"], values["C_right"]),
            R_min=min(values["R_eff_left"], values["R_eff_right"]), R_max=max(values["R_eff_left"], values["R_eff_right"]),
            boundary_count_after=after,
            shared_boundary_count=None if before is None or after is None else before - after,
            pareto_rank=_number(row.get("pareto_rank")),
            validation_rmse=_number(row.get("metrics", {}).get("validation", {}).get("rmse")),
            selected=bool(row.get("winner", False)),
        )
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame, dict(excluded)
    for target, gap in (("d_R", "pair_d_R"), ("d_C", "pair_d_C")):
        grouped = frame.groupby("stage_id")[target]
        frame[target + "_rank_percentile"] = (grouped.rank(method="average") - 0.5) / grouped.transform("size")
        frame[target + "_rank_centered"] = frame[target + "_rank_percentile"] - 0.5
        frame[target + "_stage_median"] = grouped.transform("median")
        excess = np.maximum(frame[target] ** 2 - (frame[gap] / 2) ** 2, 0)
        frame[target + "_excess"] = np.sqrt(excess)
        eg = frame.groupby("stage_id")[target + "_excess"]
        frame[target + "_excess_rank_percentile"] = (eg.rank(method="average") - 0.5) / eg.transform("size")
    frame["both_below_stage_median"] = ((frame.d_R < frame.d_R_stage_median) & (frame.d_C < frame.d_C_stage_median))
    frame["descriptive_exception"] = ~frame.both_below_stage_median
    return frame, dict(excluded)


def _selector_text(conditions):
    return " AND ".join(f"{c['feature']} {c['operator']} {c['value']:.8g}" for c in conditions)


def _mask(frame, conditions):
    mask = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        if condition["feature"] not in PREFIT_FEATURES:
            raise ValueError("Postfit or unapproved selector would introduce leakage")
        values = frame[condition["feature"]].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if condition["operator"] == "<=":
            valid &= values <= condition["value"]
        elif condition["operator"] == ">":
            valid &= values > condition["value"]
        else:
            raise ValueError("Unsupported numeric selector")
        mask &= valid
    return mask


def _eligibility(frame, mask, cfg):
    n = int(mask.sum())
    if n < cfg["min_support"]:
        return "insufficient_support"
    if n > cfg["max_coverage"] * len(frame):
        return "excessive_coverage"
    if frame.loc[mask, "stage_id"].nunique() < cfg["min_stages"]:
        return "insufficient_stages"
    if min(-frame.loc[mask, "d_R_rank_centered"].mean(), -frame.loc[mask, "d_C_rank_centered"].mean()) <= 0:
        return "no_joint_rank_improvement"
    return "eligible"


def _details(frame, conditions, rule_id, cfg, label=None):
    mask = _mask(frame, conditions)
    sub = frame.loc[mask]
    n = len(sub)
    out = dict(rule_id=rule_id, rule=label or _selector_text(conditions), conditions=conditions,
               eligibility=_eligibility(frame, mask, cfg), n=n,
               coverage=n / len(frame), unique_pairs=int(sub.pair_key.nunique()),
               n_stages=int(sub.stage_id.nunique()), trial_ids=sub.trial_id.tolist())
    if not n:
        return out
    stage_means = frame.groupby("stage_id")[["d_R", "d_C"]].mean()
    matched = stage_means.loc[sub.stage_id].mean()
    by_stage = sub.groupby("stage_id").agg(
        depth=("depth", "first"), n=("trial_id", "size"), mean_d_R=("d_R", "mean"), mean_d_C=("d_C", "mean"),
        rank_percentile_R=("d_R_rank_percentile", "mean"), rank_percentile_C=("d_C_rank_percentile", "mean"))
    er, ec = (float(-sub[t + "_rank_centered"].mean()) for t in ("d_R", "d_C"))
    # This is a fixed-rule sensitivity check, not re-fit or independent CV.
    loso = [float((-sub.loc[sub.stage_id != st, ["d_R_rank_centered", "d_C_rank_centered"]].mean()).min())
            for st in sub.stage_id.unique()] if sub.stage_id.nunique() > 1 else []
    pair_means = sub.groupby("pair_key")[["d_R_rank_percentile", "d_C_rank_percentile"]].mean().mean()
    out.update(score=math.sqrt(n / len(frame)) * min(er, ec),
               rank_improvement_R=er, rank_improvement_C=ec,
               mean_rank_percentile_R=0.5-er, mean_rank_percentile_C=0.5-ec,
               mean_d_R=float(sub.d_R.mean()), mean_d_C=float(sub.d_C.mean()),
               stage_matched_reference_d_R=float(matched.d_R), stage_matched_reference_d_C=float(matched.d_C),
               d_R_lower_percent=float(100*(1-sub.d_R.mean()/matched.d_R)) if matched.d_R else None,
               d_C_lower_percent=float(100*(1-sub.d_C.mean()/matched.d_C)) if matched.d_C else None,
               n_stages_both_improve=int(((by_stage.rank_percentile_R < .5) & (by_stage.rank_percentile_C < .5)).sum()),
               both_below_stage_median_n=int(sub.both_below_stage_median.sum()),
               descriptive_exception_n=int(sub.descriptive_exception.sum()),
               front1_n=int((sub.pareto_rank == 1).sum()),
               mean_validation_rmse=None if sub.validation_rmse.dropna().empty else float(sub.validation_rmse.mean()),
               excess_mean_rank_percentile_R=float(sub.d_R_excess_rank_percentile.mean()),
               excess_mean_rank_percentile_C=float(sub.d_C_excess_rank_percentile.mean()),
               equal_pair_mean_rank_percentile_R=float(pair_means.iloc[0]),
               equal_pair_mean_rank_percentile_C=float(pair_means.iloc[1]),
               fixed_rule_leave_stage_out_min_improvement=min(loso) if loso else None,
               by_stage=by_stage.reset_index().to_dict("records"))
    return out


def _search(frame, cfg):
    selectors = []
    for feature in PREFIT_FEATURES:
        values = np.unique(frame[feature].dropna().to_numpy(dtype=float))
        if len(values) <= 1:
            continue
        if len(values) <= 6:
            cuts = (values[:-1]+values[1:])/2
        else:
            cuts = np.unique(np.quantile(values, np.linspace(.1, .9, 9)))
            if feature.startswith("pair_d"):
                cuts = np.unique(np.r_[cuts, np.log([1.5, 2, 3, 4, 5, 10])])
        for cut in cuts:
            for op in ("<=", ">"):
                cond = dict(feature=feature, operator=op, value=float(cut))
                mask = _mask(frame, [cond])
                if cfg["min_support"] <= mask.sum() <= cfg["max_coverage"] * len(frame):
                    selectors.append((cond, mask))
    accepted = []
    rank_r = frame.d_R_rank_centered.to_numpy()
    rank_c = frame.d_C_rank_centered.to_numpy()
    stages = frame.stage_id.to_numpy()

    def consider(conditions, mask):
        n = int(mask.sum())
        if not cfg["min_support"] <= n <= cfg["max_coverage"] * len(frame):
            return
        if len(set(stages[mask])) < cfg["min_stages"]:
            return
        improvement = min(-float(rank_r[mask].mean()), -float(rank_c[mask].mean()))
        if improvement <= 0:
            return
        accepted.append((math.sqrt(n/len(frame))*improvement, conditions, mask.tobytes()))

    for cond, mask in selectors:
        consider([cond], mask)
    if cfg["max_depth"] == 2:
        for (c1, m1), (c2, m2) in itertools.combinations(selectors, 2):
            if c1["feature"] != c2["feature"]:
                consider([c1, c2], m1 & m2)
    # Stable score sort reproduces original search tie order.
    accepted.sort(key=lambda item: item[0], reverse=True)
    seen, selected = set(), []
    for _, conditions, key in accepted:
        if key in seen:
            continue
        seen.add(key)
        if len(selected) < cfg["max_rules"]:
            selected.append(_details(frame, conditions, f"N{len(selected)+1:02d}", cfg))
    return selected, len(accepted), len(seen), len(selectors)


def _serial(value):
    if isinstance(value, dict):
        return {k: _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(x) for x in value]
    if isinstance(value, np.generic):
        return _serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_serial(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _flat(rule):
    return {k: v for k, v in rule.items() if k not in {"conditions", "trial_ids", "by_stage"}}


def _report(summary, rules, readable, output):
    columns = ["rule", "eligibility", "n", "unique_pairs", "n_stages", "mean_d_R", "mean_d_C",
               "mean_rank_percentile_R", "mean_rank_percentile_C", "descriptive_exception_n"]
    def table(items):
        if not items:
            return "<p>No rule meets the configured support, stage, coverage and joint-improvement conditions.</p>"
        return pd.DataFrame([_flat(x) for x in items]).reindex(columns=columns).to_html(index=False, escape=True, float_format=lambda x:f"{x:.4g}")
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>HIBO numeric subgroup discovery</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1300px;margin:36px auto;padding:0 24px;color:#18252b}}h1,h2{{color:#164b60}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #cbd5db;text-align:left}}th{{background:#eaf0f4}}code{{background:#eef2f5}}.scroll{{overflow:auto}}a{{color:#126b99}}</style>
<h1>Steps 12–13: Numeric subgroup discovery</h1>
<p>Status: <strong>{html.escape(summary['status'])}</strong>. Eligible analysis rows: {summary['n_analysis_rows']}; unique member-set pairs: {summary['n_unique_pairs']}; parent stages: {summary['n_stages']}.</p>
<p>{html.escape(SCOPE)}</p>
<h2>Method</h2><p>Only massive–massive PASS trials enter this numeric analysis. FAIL and UNRESOLVED are not converted into new labels. Windows are separated because their C-distance convention would produce trivial comparisons. The targets are postfit d_R and d_C. Conditions use only prefit parent R/C and boundary descriptors.</p>
<p>Each target is converted to an average-rank percentile within its parent stage: (rank − 0.5)/stage size. The stage average is 0.5. Score = sqrt(n/N) × min(0.5 − mean R percentile, 0.5 − mean C percentile). Both improvements must be positive. This is a declared exploratory analysis choice, not a universal standard.</p>
<p>Configuration: <code>{html.escape(json.dumps(summary['config']))}</code>. Search thresholds come from prefit quantiles and fixed ratio values. Membership duplicates are removed. Stage, RMSE, convergence, Pareto rank and postfit values are not rule predictors.</p>
<h2>Readable candidate rules</h2><p>The following previously discussed cutoffs are re-evaluated on this input, even when they do not meet the rule criteria. They are explanatory examples, not fixed physical thresholds. The eligibility column must be checked.</p><div class="scroll">{table(readable)}</div>
<h2>Top automatic search rules</h2><div class="scroll">{table(rules)}</div>
<h2>Evidence and limits</h2><p>Raw mean-distance comparisons use a reference weighted to match the subgroup's parent-stage composition. A descriptive exception means that at least one postfit distance is not below its stage median; this is not a physical FAIL or the rule-search target. Member records include supporting cases and exceptions. Different rules can overlap heavily.</p>
<p>{html.escape(DISTANCE_NOTE)}</p><p>Equal-pair weighting, nearby thresholds and fixed-rule leave-one-stage-out calculations are same-data sensitivity checks, not independent validation. New setpoints, periods, buildings or reduction paths are needed to assess transfer. Never skip MCMC/convergence/RMSE checks based on these rules.</p>
<h2>Reproducible outputs</h2><ul>'''
    for filename in ("summary.json", "numeric_rules.json", "numeric_rules.csv", "readable_rules.json", "readable_rules.csv", "rule_members.csv", "analysis_rows.csv", "threshold_sensitivity.csv"):
        text += f'<li><a href="{filename}">{filename}</a></li>'
    text += "</ul></html>"
    (output / "report.html").write_text(text, encoding="utf-8")


def discover_numeric_rules(rows, config=None, output=None):
    """Write numeric subgroup results and return their summary.
    
    ``rows`` contains one authoritative result per trial, using the latest
    predeclared attempt. ``output`` specifies the destination directory.
    Input labels and records are preserved."""
    cfg = _config(config)
    source = list(rows)
    frame, excluded = _prepare(source)
    output = Path(output or "numeric_rules")
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(schema="hibo-numeric-subgroups-v1", config=cfg, scope=SCOPE,
                   distance_definition_note=DISTANCE_NOTE, prefit_features=list(PREFIT_FEATURES),
                   targets=["postfit d_R", "postfit d_C"], outcome_counts=dict(Counter(str(r.get("status", "")).lower() for r in source)),
                   n_input_rows=len(source), n_analysis_rows=len(frame), excluded_counts=excluded,
                   n_unique_pairs=0 if frame.empty else int(frame.pair_key.nunique()),
                   n_stages=0 if frame.empty else int(frame.stage_id.nunique()),
                   rule_feedback_into_reduction=False, new_mcmc_performed=False)
    rules, readable, sensitivity = [], [], []
    if frame.empty or len(frame) < cfg["min_support"] or frame.stage_id.nunique() < cfg["min_stages"]:
        summary.update(status="insufficient_data", n_search_candidates=0, n_unique_membership_rules=0, n_selectors=0)
    else:
        rules, attempted, distinct, selectors = _search(frame, cfg)
        summary.update(status="rules_found" if rules else "no_joint_improvement_rules", n_search_candidates=attempted,
                       n_unique_membership_rules=distinct, n_selectors=selectors)
        summary["baseline"] = {"mean_d_R": float(frame.d_R.mean()), "mean_d_C": float(frame.d_C.mean()),
                               "mean_rank_percentile_R": float(frame.d_R_rank_percentile.mean()),
                               "mean_rank_percentile_C": float(frame.d_C_rank_percentile.mean())}
        for rid, label, conditions in (
            ("E01", "R_eff ratio <= 3", [dict(feature="pair_d_R", operator="<=", value=math.log(3))]),
            ("E02", "R_eff ratio <= 4 AND C ratio <= 4", [dict(feature=f, operator="<=", value=math.log(4)) for f in ("pair_d_R", "pair_d_C")]),
        ):
            readable.append(_details(frame, conditions, rid, cfg, label))
        for ratio in (2.5, 2.75, 3, 3.25, 3.5, 4):
            detail = _details(frame, [dict(feature="pair_d_R", operator="<=", value=math.log(ratio))], f"S{ratio}", cfg)
            sensitivity.append(dict(R_ratio_threshold=ratio, **_flat(detail)))
        summary["readable_rule_overlap_n"] = len(set(readable[0]["trial_ids"]) & set(readable[1]["trial_ids"]))
    summary["n_rules_reported"] = len(rules)
    summary["readable_rules"] = [_flat(rule) for rule in readable]
    members = []
    for rule in rules + readable:
        selected = frame[frame.trial_id.isin(rule["trial_ids"])]
        for record in selected.to_dict("records"):
            members.append(dict(rule_id=rule["rule_id"], rule=rule["rule"], **record))
    _write_json(output / "numeric_rules.json", rules)
    _write_json(output / "readable_rules.json", readable)
    for filename, records, fallback in (
        ("numeric_rules.csv", [_flat(x) for x in rules], ["rule_id", "rule", "n", "score"]),
        ("readable_rules.csv", [_flat(x) for x in readable], ["rule_id", "rule", "eligibility", "n"]),
        ("rule_members.csv", members, ["rule_id", "trial_id", "descriptive_exception"]),
        ("threshold_sensitivity.csv", sensitivity, ["R_ratio_threshold", "n", "eligibility"]),
    ):
        data = pd.DataFrame(records) if records else pd.DataFrame(columns=fallback)
        data.to_csv(output / filename, index=False)
    (frame if not frame.empty else pd.DataFrame(columns=["trial_id", "depth", "stage_id", *PREFIT_FEATURES, "d_R", "d_C"])).to_csv(output / "analysis_rows.csv", index=False)
    _write_json(output / "summary.json", summary)
    _report(summary, rules, readable, output)
    return _serial(summary)
