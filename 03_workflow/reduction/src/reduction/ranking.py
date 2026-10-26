"""Postfit parameter-change objectives and Pareto candidate ranking.

Each objective compares the fitted merged block in M(k+1) with its two
constituents in M(k). Changes include both structural aggregation and
subsequent refitting: summed capacitances and parallel conductances can
differ from constituent values even before refitting."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from .topology import BOUNDARIES, direct_ports, merge_pair, validate_model


def _positive(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _r_eff(block: Mapping) -> float:
    if block["kind"] in ("window", "massless"):
        return 1.0 / sum(1.0 / _positive(r, "direct R") for r in direct_ports(block) if r is not None)
    resistances = [_positive(r, "boundary R") for r in block["r_boundary"] if r is not None]
    if not resistances:
        raise ValueError("Massive block has no active boundary port")
    return _positive(block["r_air"], "air-side R") + 1.0 / sum(1.0 / r for r in resistances)


def _distance(after: float, before: float) -> float:
    # Separate logarithms avoid overflow/underflow of a ratio at extreme scales.
    return abs(math.log(_positive(after, "after")) - math.log(_positive(before, "before")))


def _merged_block(model: Mapping, members: set[str]) -> Mapping:
    matching = [b for b in model["blocks"] if set(b["members"]) == members]
    if len(matching) != 1:
        raise ValueError("Child must contain exactly one block with the union of the two parent member sets")
    return matching[0]


def parameter_changes(parent: dict, fitted_child: dict, left_id: str, right_id: str) -> dict:
    """Compute literal merged-versus-parent changes after the candidate fit.

    d_R = sqrt(mean_i ln(R_eff,merged / R_eff,parent_i)**2) for both constituents.
    d_C = sqrt(mean_i ln(C_merged / C_parent_i)**2) for massive constituents only.
    For a massless/massive pair, C is compared with the massive block alone;
    log(0) is never evaluated. A purely massless pair has no C comparison and
    uses d_C=0 by an explicit no-C-change convention. R_eff equals R_air plus
    parallel boundary R for a massive block, and parallel direct boundary R
    for a massless block. R_eff is a scalar descriptor,
    not a claim that distinct boundary temperature inputs have been collapsed.

    Root-mean-square log ratios avoid the flat interval produced by averaging
    absolute log ratios between two parent values, and give greater influence
    to a larger constituent deviation without introducing adjustable weights.
    Portwise changes and deviations from the physical aggregation initializer
    are supplementary diagnostics and do not enter Pareto ranking.
    """
    validate_model(parent)
    validate_model(fitted_child)
    if left_id == right_id:
        raise ValueError("Two distinct parent blocks are required")
    by_id = {b["id"]: b for b in parent["blocks"]}
    try:
        constituents = [by_id[left_id], by_id[right_id]]
    except KeyError as exc:
        raise KeyError("A constituent block is missing from the parent") from exc
    members = set(constituents[0]["members"]) | set(constituents[1]["members"])
    fitted = _merged_block(fitted_child, members)
    initialized_child, _, _ = merge_pair(parent, left_id, right_id)
    initialized = _merged_block(initialized_child, members)
    new_r = _r_eff(fitted)
    fitted_massive = fitted["kind"] == "massive"
    new_c = _positive(fitted["C"], "fitted merged C") if fitted_massive else 0.0
    per_parent, portwise = [], []
    r_changes, c_changes = [], []
    for block in constituents:
        old_r = _r_eff(block)
        r_change = _distance(new_r, old_r)
        r_changes.append(r_change)
        massive = block["kind"] == "massive"
        old_c = _positive(block["C"], "parent C") if massive else 0.0
        c_change = _distance(new_c, old_c) if massive else None
        if massive:
            if not fitted_massive:
                raise ValueError("A massive constituent cannot lose its thermal state in a massless merge")
            c_changes.append(c_change)
        per_parent.append({
            "parent_block_id": block["id"], "parent_members": list(block["members"]),
            "parent_kind": block["kind"], "R_eff_parent_K_per_W": old_r,
            "R_eff_fitted_merged_K_per_W": new_r, "abs_log_R_change": r_change,
            "C_parent_J_per_K": old_c, "C_fitted_merged_J_per_K": new_c,
            "C_comparison_applicable": massive, "abs_log_C_change": c_change,
            "C_comparison_note": "Compared with massive constituent" if massive else
                "Not applicable: massless constituent has C=0; no log(0) or artificial capacitance",
        })
        if massive:
            portwise.append({
                "parent_block_id": block["id"], "port": "air", "applicable": True,
                "parent_R_K_per_W": float(block["r_air"]),
                "fitted_merged_R_K_per_W": float(fitted["r_air"]),
                "abs_log_change": _distance(fitted["r_air"], block["r_air"]),
            })
            for boundary, old, new in zip(BOUNDARIES, block["r_boundary"], fitted["r_boundary"]):
                if old is not None and new is None:
                    raise ValueError("A merged block lost an active constituent boundary port")
                if old is None and new is None:
                    continue
                portwise.append({
                    "parent_block_id": block["id"], "port": boundary,
                    "applicable": old is not None,
                    "parent_R_K_per_W": None if old is None else float(old),
                    "fitted_merged_R_K_per_W": None if new is None else float(new),
                    "abs_log_change": None if old is None else _distance(new, old),
                    "note": "Same active port" if old is not None else
                        "Parent had no branch at this boundary; no zero/infinite R substituted",
                })
        else:
            for boundary, old in zip(BOUNDARIES, direct_ports(block)):
                if old is None:
                    continue
                new = None if fitted_massive else direct_ports(fitted)[BOUNDARIES.index(boundary)]
                portwise.append({
                    "parent_block_id": block["id"], "port": "direct_to_air",
                    "boundary": boundary, "applicable": not fitted_massive,
                    "parent_R_K_per_W": float(old),
                    "fitted_merged_R_K_per_W": new,
                    "abs_log_change": None if fitted_massive else _distance(new, old),
                    "note": "Direct branch absorbed into massive block; no one-to-one merged port" if fitted_massive else
                        "Direct conductance aggregated at the same boundary; other boundary inputs remain separate",
                })
    parent_c_sum = sum(float(b["C"]) for b in constituents)
    initialized_r = _r_eff(initialized)
    return {
        "schema": "hibo_postfit_parameter_changes_allpairs_v2", "stage": "postfit",
        "left_id": left_id, "right_id": right_id,
        "merged_block_id": fitted["id"], "merged_members": sorted(members),
        "window_pair": any(b["kind"] in ("window", "massless") for b in constituents),
        "d_R": math.hypot(*r_changes) / math.sqrt(len(r_changes)),
        "d_C": math.hypot(*c_changes) / math.sqrt(len(c_changes)) if c_changes else 0.0,
        "C_objective_applicable": bool(c_changes),
        "C_objective_note": "RMS log ratio over massive constituents" if c_changes else
            "Not applicable: both constituents and merged block are massless; zero is a neutral no-C-change convention, not log(0)",
        "R_comparison_count": len(r_changes), "C_comparison_count": len(c_changes),
        "per_parent": per_parent, "portwise_R_changes": portwise,
        "supplementary": {
            "used_for_ranking": False,
            "parent_C_sum_J_per_K": parent_c_sum,
            "fitted_merged_C_J_per_K": new_c,
            "merged_to_parent_C_sum_ratio": new_c / parent_c_sum if parent_c_sum else None,
            "abs_log_C_change_from_parent_sum": _distance(new_c, parent_c_sum) if parent_c_sum else None,
            "physical_aggregate_initial_R_eff_K_per_W": initialized_r,
            "fitted_merged_R_eff_K_per_W": new_r,
            "abs_log_R_refit_change_from_physical_aggregate": _distance(new_r, initialized_r),
            "physical_aggregate_initial_C_J_per_K": float(initialized["C"]),
            "abs_log_C_refit_change_from_physical_aggregate": _distance(new_c, initialized["C"]) if fitted_massive else None,
            "definition": "Physical aggregation initializer computed from the fitted parent before candidate refitting; not the original-IDF prior mean",
        },
        "definition": {
            "R_eff": "R_air + 1/sum(1/R_boundary) for massive blocks; 1/sum(1/R_direct) for massless blocks",
            "d_R": "Root mean square of two natural-log ratios: fitted merged R_eff versus each parent constituent R_eff",
            "d_C": "Root mean square of natural-log ratios versus massive constituents; massless constituents excluded. Purely massless merges use zero by explicit no-C-change convention",
            "aggregation_note": "RMS log ratios avoid the flat interval of the mean absolute-log distance between two parent values; all constituent comparisons have equal weight",
            "interpretation": "Literal postfit changes relative to each constituent, not premerge similarity and not a pure refit penalty",
            "structural_change_note": "Parallel conductance aggregation and capacitance summation can make d_R and d_C nonzero even without refitting",
            "boundary_note": "Scalar R_eff is a descriptor; actual simulation retains each active boundary port and temperature input",
        },
    }


def _finite_nonnegative(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative for every PASS trial")
    return value


def rank_trials(rows: list[Mapping], min_pass_candidates: int = 1) -> list[dict]:
    """Rank PASS candidates and select a first-front member.
    
    Both postfit d_R and d_C are minimized. Dominance requires no worse values
    in either objective and a strictly better value in at least one. First-front
    ties are resolved by validation RMSE, child n_C, child n_R, and trial_id.
    A single PASS candidate is eligible. Non-PASS rows retain pareto_rank=None
    and winner=False. The compatibility argument must equal one. The input
    records are not modified."""
    if isinstance(min_pass_candidates,bool) or min_pass_candidates != 1:
        raise ValueError("Step 10 requires min_pass_candidates=1: a sole PASS must be adopted")
    result = [deepcopy(dict(row)) for row in rows]
    ids = [str(row["trial_id"]) for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate trial_id prevents deterministic ranking")
    points: dict[int, tuple[float, float]] = {}
    tie_breaks = {}
    for idx, row in enumerate(result):
        row["pareto_rank"] = None
        row["winner"] = False
        if str(row.get("status", "")).lower() != "pass":
            continue
        changes = row.get("parameter_changes", row.get("changes"))
        if not isinstance(changes, Mapping):
            raise ValueError(f"PASS trial {ids[idx]} requires postfit parameter_changes")
        if changes.get("stage", "postfit") != "postfit":
            raise ValueError("Ranking changes must be postfit, not premerge pair descriptors")
        points[idx] = (_finite_nonnegative(changes["d_R"], "d_R"),
                       _finite_nonnegative(changes["d_C"], "d_C"))
        rmse = _finite_nonnegative(row["metrics"]["validation"]["rmse"], "validation RMSE")
        features = row["features"]
        n_c, n_r = features["child_n_C"], features["child_n_R"]
        if any(isinstance(x, bool) or not math.isfinite(float(x)) or float(x) <= 0 or float(x) != int(x)
               for x in (n_c, n_r)):
            raise ValueError("PASS trials require positive integer child_n_C and child_n_R")
        tie_breaks[idx] = (rmse, int(n_c), int(n_r), ids[idx])
    comparison = dict(n_pass_candidates=len(points), min_pass_candidates=1,
                      adoption_ready=bool(points),
                      comparative_selection_ready=len(points)>=2,
                      reason="multiple_passing_candidates" if len(points)>=2 else
                          "sole_passing_candidate" if points else "no_passing_candidate")
    for row in result:
        row["comparison"] = deepcopy(comparison)
    remaining = set(points)
    front_number = 1
    while remaining:
        front = [i for i in sorted(remaining) if not any(
            j != i and points[j][0] <= points[i][0] and points[j][1] <= points[i][1]
            and (points[j][0] < points[i][0] or points[j][1] < points[i][1])
            for j in remaining
        )]
        if not front:  # Finite points guarantee a nonempty nondominated front.
            raise RuntimeError("No Pareto front found for finite objectives")
        for idx in front:
            result[idx]["pareto_rank"] = front_number
        if front_number == 1 and len(points)>=min_pass_candidates:
            result[min(front, key=lambda i: tie_breaks[i])]["winner"] = True
        remaining.difference_update(front)
        front_number += 1
    return result
