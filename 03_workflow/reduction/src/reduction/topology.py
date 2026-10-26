"""All-pairs reduction of the calibrated HIBO Lab B envelope network.

The air state is retained throughout reduction. Any two envelope blocks may
merge, with each boundary temperature preserved as a separate input.
Aggregation supplies initial values for subsequent estimation of the merged
block's R, C, and source allocations. Each merged allocation shares a fixed
budget with zone air. Massless pairs retain direct boundary-to-air ports
without adding a thermal state or source allocation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

import numpy as np

BOUNDARIES = ("plenum", "outdoor", "neighbor102plus", "ground")
HEAT_SOURCES = ("solar", "lighting", "occupant", "equipment")
SCHEMA = "hibo_reduction_network_allpairs_rca_v2"


def direct_ports(block: dict, *, reference: bool = False) -> list[float | None]:
    """Four boundary-to-air R slots for a singleton or grouped massless block."""
    fields = block["reference"] if reference else block
    if block["kind"] == "window":
        ports = [None] * 4
        boundary = block["boundary"]
        if isinstance(boundary, bool) or boundary not in range(4):
            raise ValueError("Invalid window boundary")
        ports[boundary] = fields["R"]
        return ports
    if block["kind"] == "massless":
        return list(fields["r_direct"])
    raise ValueError("Direct ports belong only to massless blocks")


def _positive(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _estimate(estimates: dict, key: str) -> float:
    if key not in estimates:
        raise KeyError(f"Representative joint draw lacks {key}")
    item = estimates[key]
    return float(item["base_value"] if isinstance(item, dict) else item)


def _touch(vertices_a: Any, vertices_b: Any) -> bool:
    """Detect a positive-length collinear shared edge in metre coordinates.
    
    A shared vertex is insufficient. Tolerances follow prepare_reference.py."""
    a, b = np.asarray(vertices_a, dtype=float), np.asarray(vertices_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("Surface vertices must be n-by-3 coordinates")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Surface vertices must be finite")
    for p, q in zip(a, np.roll(a, -1, axis=0)):
        u = q - p
        norm = float(np.linalg.norm(u))
        if norm <= 1e-12:
            continue
        for r, s in zip(b, np.roll(b, -1, axis=0)):
            if np.linalg.norm(s - r) <= 1e-12:
                continue
            if np.linalg.norm(np.cross(u, s-r)) < 1e-5 and np.linalg.norm(np.cross(u, r-p)) < 1e-5:
                t0, t1 = sorted((float(np.dot(r-p, u)/norm), float(np.dot(s-p, u)/norm)))
                if min(norm, t1) - max(0.0, t0) > 1e-4:
                    return True
    return False


def _window_hosts(reference: dict) -> dict[str, str]:
    components = reference["components"]
    hosts = {}
    for comp in components:
        if comp["kind"] not in ("massless", "window"):
            continue
        host = comp.get("host")
        if isinstance(host, int):
            if host < 0 or host >= len(components):
                raise ValueError(f"Invalid window host index: {comp['name']}")
            host = components[host]["name"]
        if not isinstance(host, str) or host not in {c["name"] for c in components if c["kind"] == "massive"}:
            raise ValueError(f"Window has no valid original opaque host: {comp['name']}")
        hosts[comp["name"]] = host
    return hosts


def adjacency_from_reference(reference: dict) -> list[tuple[str, str]]:
    """Return geometric adjacency between the initial components.
    
    The HIBO reference contains 15 opaque adjacency edges and three window-host
    edges. Adjacency is descriptive metadata, independent of fitting."""
    opaque = [c for c in reference["components"] if c["kind"] == "massive"]
    edges = set()
    for i, a in enumerate(opaque):
        for b in opaque[i+1:]:
            if any(_touch(x, y) for x in a["vertices"] for y in b["vertices"]):
                edges.add(tuple(sorted((a["name"], b["name"]))))
    for window, host in _window_hosts(reference).items():
        edges.add(tuple(sorted((window, host))))
    return sorted(edges)


def validate_model(model: dict) -> None:
    """Reject invalid physical parameters and duplicated/lost original objects."""
    ids, members = set(), []
    gain_totals = np.asarray(model["air"]["alpha"], dtype=float).copy()
    if gain_totals.shape != (4,) or not np.isfinite(gain_totals).all() or np.any(gain_totals < 0):
        raise ValueError("Air alpha must contain four nonnegative finite fractions")
    _positive(model["air"]["C"], "air C")
    _positive(model["noise"]["sigma_hour"], "sigma_hour")
    rho = float(model["noise"]["rho_hour"])
    if not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("rho_hour must be between zero and one")
    original_massive = set(model.get("provenance", {}).get("original_massive_members", []))
    for block in model["blocks"]:
        if block["id"] in ids:
            raise ValueError("Duplicate block ID")
        ids.add(block["id"])
        if not block["members"] or len(block["members"]) != len(set(block["members"])):
            raise ValueError("Each block must contain distinct original members")
        members.extend(block["members"])
        alpha = np.asarray(block["alpha"], dtype=float)
        if alpha.shape != (4,) or not np.isfinite(alpha).all() or np.any(alpha < 0):
            raise ValueError("Block alpha must contain four nonnegative finite fractions")
        gain_totals += alpha
        area = float(block["area"])
        if not math.isfinite(area) or area <= 0:
            raise ValueError("Block area must be positive")
        if block["kind"] == "massive":
            for fields in (block, block["reference"]):
                ports = fields["r_boundary"]
                if len(ports) != 4 or all(r is None for r in ports):
                    raise ValueError("Massive block requires four slots and an active boundary port")
                for r in ports:
                    if r is not None:
                        _positive(r, "boundary R")
                _positive(fields["r_air"], "air-side R")
                _positive(fields["C"], "block C")
            if [r is not None for r in block["r_boundary"]] != [r is not None for r in block["reference"]["r_boundary"]]:
                raise ValueError("Fitted and IDF-reference boundary ports must match")
            if original_massive and not original_massive.intersection(block["members"]):
                raise ValueError("A massive block must contain an original massive component")
        elif block["kind"] in ("window", "massless"):
            if block["kind"] == "window" and len(block["members"]) != 1:
                raise ValueError("A singleton window must have exactly one member")
            if original_massive.intersection(block["members"]):
                raise ValueError("A massless block cannot contain an original massive component")
            fitted_ports, ref_ports = direct_ports(block), direct_ports(block, reference=True)
            for ports in (fitted_ports, ref_ports):
                if len(ports) != 4 or all(r is None for r in ports):
                    raise ValueError("A massless block needs four slots and an active direct port")
                for r in ports:
                    if r is not None:
                        _positive(r, "direct R")
            if [r is not None for r in fitted_ports] != [r is not None for r in ref_ports]:
                raise ValueError("Fitted and IDF-reference direct ports must match")
            if float(block["C"]) != 0 or np.any(alpha != 0):
                raise ValueError("Massless blocks cannot carry capacitance or heat allocation")
        else:
            raise ValueError(f"Unknown block kind: {block['kind']}")
    if len(members) != len(set(members)):
        raise ValueError("Original object appears in more than one block")
    original = model.get("provenance", {}).get("original_members")
    if original is not None and set(members) != set(original):
        raise ValueError("A reduction lost or introduced an original object")
    if not np.allclose(gain_totals, 1.0, rtol=0, atol=1e-8):
        raise ValueError("Heat allocation fractions must sum to one for every source")


def build_initial_model(reference: dict, estimates_by_parameter: dict) -> dict:
    """Construct M0 from one coherent representative posterior draw.
    
    Prior means are taken from the IDF reference. Parameter units are K/W, J/K,
    and dimensionless heat-source fractions."""
    hosts = _window_hosts(reference)
    blocks = []
    for comp in reference["components"]:
        name, boundary = comp["name"], int(comp["boundary"])
        if boundary not in range(4):
            raise ValueError("Unknown boundary in reference")
        if comp["kind"] == "massive":
            ports, ref_ports = [None]*4, [None]*4
            ports[boundary] = _estimate(estimates_by_parameter, name+":R1")
            ref_ports[boundary] = float(comp["R1"])
            block = dict(id=name, members=[name], kind="massive", r_boundary=ports,
                         r_air=_estimate(estimates_by_parameter, name+":R2"),
                         C=_estimate(estimates_by_parameter, name+":C"),
                         alpha=[_estimate(estimates_by_parameter, f"alpha_{source}:{name}") for source in HEAT_SOURCES],
                         area=float(comp["area"]),
                         reference=dict(r_boundary=ref_ports, r_air=float(comp["R2"]), C=float(comp["C"])))
        elif comp["kind"] in ("massless", "window"):
            block = dict(id=name, members=[name], kind="window", boundary=boundary,
                         R=_estimate(estimates_by_parameter, name+":R"), C=0.0,
                         alpha=[0.0]*4, area=float(comp["area"]), reference={"R":float(comp["R"])})
        else:
            raise ValueError(f"Unknown component kind: {comp['kind']}")
        blocks.append(block)
    model = dict(schema=SCHEMA, depth=0, blocks=blocks,
                 air=dict(C=_estimate(estimates_by_parameter, "zone_air:C"),
                          alpha=[_estimate(estimates_by_parameter, f"alpha_{source}:zone_air") for source in HEAT_SOURCES]),
                 noise=dict(sigma_hour=_estimate(estimates_by_parameter, "sigma_innovation_hour"),
                            rho_hour=_estimate(estimates_by_parameter, "rho_hour")),
                 provenance=dict(target_zone=reference.get("target_zone"), boundary_order=list(BOUNDARIES),
                                 original_members=[c["name"] for c in reference["components"]],
                                 original_massive_members=[c["name"] for c in reference["components"] if c["kind"] == "massive"],
                                 candidate_policy="all_unordered_envelope_block_pairs",
                                 geometry_metadata_role="descriptive_only_not_a_merge_constraint",
                                 window_hosts=hosts, adjacency=[list(e) for e in adjacency_from_reference(reference)],
                                 baseline="Case 5 exported representative joint posterior draw",
                                 prior_reference="Original IDF-derived references, never fitted-parent values",
                                 merge_history=[]))
    validate_model(model)
    return model


def candidate_pairs(model: dict, adjacency: list | tuple | None = None) -> list[tuple[str, str]]:
    """Return all n(n-1)/2 pairs of envelope blocks.
    
    Zone air is outside ``blocks``. Geometry and window-host relationships are
    retained as metadata and do not restrict candidate eligibility."""
    validate_model(model)
    blocks = sorted(model["blocks"], key=lambda b:b["id"])
    return [(a["id"], b["id"]) for i, a in enumerate(blocks) for b in blocks[i+1:]]


def counts(model: dict) -> dict:
    massive = [b for b in model["blocks"] if b["kind"] == "massive"]
    windows = [b for b in model["blocks"] if b["kind"] in ("window", "massless")]
    nr = sum(1+sum(r is not None for r in b["r_boundary"]) for b in massive)
    nr += sum(sum(r is not None for r in direct_ports(b)) for b in windows)
    nc = len(massive)+1
    return dict(n_R=nr, n_C=nc, n_blocks=len(model["blocks"]), n_massive=len(massive),
                n_windows=len(windows), n_massless=len(windows), label=f"{nr}R{nc}C")


def _parallel(values: list) -> float | None:
    active = [float(r) for r in values if r is not None]
    if len(active) == 1:
        return active[0]
    return 1.0/sum(1.0/r for r in active) if active else None


def _aggregate_opaque(a: dict, b: dict) -> dict:
    return dict(r_boundary=[_parallel([x, y]) for x, y in zip(a["r_boundary"], b["r_boundary"])],
                r_air=_parallel([a["r_air"], b["r_air"]]), C=float(a["C"]+b["C"]))


def _absorb_massless(massive: dict, direct: list) -> dict:
    """DC-matched initialization for any massless/massive pairing.

    Original hosting does not constrain the operator. Every direct boundary
    conductance is retained, including boundaries absent from the massive block.
    """
    gb = np.array([0.0 if r is None else 1.0/r for r in massive["r_boundary"]], dtype=float)
    gz = 1.0/massive["r_air"]
    total = float(gb.sum()+gz)
    f, one_minus_f = gz/total, float(gb.sum())/total
    k_new = f*gb + np.array([0.0 if r is None else 1.0/r for r in direct])
    new_gb = k_new/f
    new_gz = float(k_new.sum())/one_minus_f
    return dict(r_boundary=[None if g == 0 else float(1.0/g) for g in new_gb],
                r_air=1.0/new_gz, C=float(massive["C"]))


def _r_eff(block: dict) -> float:
    if block["kind"] in ("window", "massless"):
        return float(1.0/sum(1.0/r for r in direct_ports(block) if r is not None))
    return float(block["r_air"] + 1.0/sum(1.0/r for r in block["r_boundary"] if r is not None))


def _active_boundary_count(block: dict) -> int:
    ports = block["r_boundary"] if block["kind"] == "massive" else direct_ports(block)
    return sum(r is not None for r in ports)


def model_hash(model: dict) -> str:
    """Stable exact-network identity, including fixed values and provenance."""
    encoded = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def merge_pair(model: dict, left_id: str, right_id: str) -> tuple[dict, list[dict], dict]:
    """Initialize any pair and declare the merged block's free R/C/alpha.

    All nonmerged envelope parameters, air C and noise remain fixed. For a
    massive merged block, each source's merged-plus-air alpha budget is fixed
    while its split is estimated. Pure massless pairs have R only: C and alpha
    remain structurally absent. The parent is never mutated.
    """
    validate_model(model)
    if left_id == right_id:
        raise ValueError("Cannot merge a block with itself")
    by_id = {b["id"]:b for b in model["blocks"]}
    if left_id not in by_id or right_id not in by_id:
        raise KeyError("Merge refers to a missing block")
    a, b = by_id[left_id], by_id[right_id]
    massive_a, massive_b = a["kind"] == "massive", b["kind"] == "massive"
    window_pair = not (massive_a and massive_b)
    members = sorted(a["members"]+b["members"])
    block_id = "merged_"+hashlib.sha256("\0".join(members).encode()).hexdigest()[:16]
    if massive_a and massive_b:
        values, ref_values = _aggregate_opaque(a, b), _aggregate_opaque(a["reference"], b["reference"])
        kind, operator = "massive", "opaque_state_aggregation"
        approximation = ("Approximation: initialize by summing C, alpha and conductances per boundary and air port. "
                         "No physical-contact restriction; R/C/alpha are subsequently estimated.")
    elif massive_a or massive_b:
        massive, massless = (a, b) if massive_a else (b, a)
        values = _absorb_massless(massive, direct_ports(massless))
        ref_values = _absorb_massless(massive["reference"], direct_ports(massless, reference=True))
        kind, operator = "massive", "massless_massive_dc_matched_absorption"
        approximation = ("Approximation: remove the direct branch once and preserve its initial steady-state "
                         "boundary-to-zone coefficients. No original-host restriction; transient equivalence is not asserted.")
    else:
        values = dict(r_direct=[_parallel([x,y]) for x,y in zip(direct_ports(a),direct_ports(b))], C=0.0)
        ref_values = dict(r_direct=[_parallel([x,y]) for x,y in zip(direct_ports(a,reference=True),direct_ports(b,reference=True))])
        kind, operator = "massless", "massless_direct_port_aggregation"
        approximation = ("Exact initial direct-conductance aggregation by boundary. Distinct boundary inputs "
                         "remain separate; grouping may reduce component count without reducing R count. No C or alpha invented.")
    merged = dict(id=block_id, members=members, kind=kind, **values,
                  alpha=[float(x+y) for x, y in zip(a["alpha"], b["alpha"])],
                  area=float(a["area"]+b["area"]), reference=ref_values)
    child = deepcopy(model)
    child["depth"] = int(model.get("depth", 0))+1
    child["blocks"] = sorted([c for c in child["blocks"] if c["id"] not in (left_id, right_id)]+[merged], key=lambda c:c["id"])
    child["provenance"].setdefault("merge_history", []).append(dict(
        depth=child["depth"], parent_hash=model_hash(model), left_id=left_id, right_id=right_id,
        new_block_id=block_id, members=members, operator=operator, approximation=approximation))
    validate_model(child)
    specs = []
    port_field = "r_boundary" if kind == "massive" else "r_direct"
    for k, r in enumerate(merged[port_field]):
        if r is not None:
            specs.append(dict(name=f"{block_id}:R_{BOUNDARIES[k]}", block_id=block_id,
                              field=port_field, index=k, transform="log", initial=float(r),
                              prior_mean=float(ref_values[port_field][k]), prior_sd=2*float(ref_values[port_field][k])))
    if kind == "massive":
        for field, suffix in (("r_air", "R_air"), ("C", "C")):
            specs.append(dict(name=f"{block_id}:{suffix}", block_id=block_id, field=field, transform="log",
                              initial=float(merged[field]), prior_mean=float(ref_values[field]), prior_sd=2*float(ref_values[field])))
        original_massive = set(model["provenance"]["original_massive_members"])
        prior_a = float(len(original_massive.intersection(members)))
        if prior_a <= 0:
            raise ValueError("A massive merge must retain an original massive state")
        for source_index, source in enumerate(HEAT_SOURCES):
            budget = float(merged["alpha"][source_index]+model["air"]["alpha"][source_index])
            if budget == 0.0:
                continue  # Structurally zero fixed pool has no identifiable split.
            fraction = float(merged["alpha"][source_index])/budget
            initial_fraction = min(max(fraction, 1e-8), 1.0-1e-8)
            specs.append(dict(name=f"{block_id}:alpha_{source}", block_id=block_id,
                              field="alpha_fraction", index=source_index, transform="logit",
                              initial=budget*initial_fraction, budget=budget, prior_a=prior_a, prior_b=1.0,
                              prior_mean=budget*prior_a/(prior_a+1.0),
                              prior_sd=budget*math.sqrt(prior_a/((prior_a+1.0)**2*(prior_a+2.0))),
                              initialization_note="Endpoint fractions, if present, move to 1e-8 or 1-1e-8 for initialization only; target is not clipped",
                              prior_definition="Conditional aggregate Dirichlet(1,...,1) split between original massive members and zone air"))
    parent_count, child_count = counts(model), counts(child)
    reff_a, reff_b = _r_eff(a), _r_eff(b)
    features = dict(left_id=left_id, right_id=right_id, left_members=list(a["members"]),
                    right_members=list(b["members"]), pair_d_R=abs(math.log(reff_a/reff_b)),
                    pair_d_C=None if window_pair else abs(math.log(a["C"]/b["C"])),
                    R_eff_left=reff_a, R_eff_right=reff_b, C_left=float(a["C"]), C_right=float(b["C"]),
                    window_pair=window_pair, operator=operator,
                    boundary_count_before=_active_boundary_count(a)+_active_boundary_count(b),
                    boundary_count_after=_active_boundary_count(merged),
                    parent_n_R=parent_count["n_R"], parent_n_C=parent_count["n_C"],
                    child_n_R=child_count["n_R"], child_n_C=child_count["n_C"],
                    n_merged_original_members=len(members), parent_depth=int(model.get("depth", 0)),
                    dimension_reduced=child_count["n_R"]<parent_count["n_R"] or child_count["n_C"]<parent_count["n_C"],
                    thermal_order_reduced=child_count["n_R"]<parent_count["n_R"] or child_count["n_C"]<parent_count["n_C"],
                    delta_n_R=child_count["n_R"]-parent_count["n_R"], delta_n_C=child_count["n_C"]-parent_count["n_C"],
                    n_free_thermal=sum(s["transform"]=="log" for s in specs),
                    n_free_alpha=sum(s["transform"]=="logit" for s in specs))
    return child, specs, features


def initial_parameter_coordinates(free_specs: list[dict]) -> np.ndarray:
    """Mixed unconstrained coordinates: log R/C and logit(alpha / budget)."""
    coordinates=[]
    for spec in free_specs:
        value = _positive(spec["initial"], "initial parameter")
        if spec.get("transform", "log") == "logit":
            budget = _positive(spec["budget"], "alpha budget")
            if value >= budget:
                raise ValueError("Alpha initializer must be strictly inside its budget")
            coordinates.append(math.log(value)-math.log(budget-value))
        elif spec.get("transform", "log") == "log":
            coordinates.append(math.log(value))
        else:
            raise ValueError("Unknown parameter transform")
    return np.asarray(coordinates, dtype=float)


def apply_parameters(model: dict, free_specs: list[dict], log_values: Any) -> dict:
    """Update declared merged R/C/alpha in mixed log/logit coordinates.

    The corresponding air alpha is a derived complement, not an independently
    fitted parameter. Nonmerged envelope R/C/alpha and air C/noise are fixed.
    """
    coordinates = np.asarray(log_values, dtype=float)
    if coordinates.ndim != 1 or len(coordinates) != len(free_specs) or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite and match the free specification count")
    child = deepcopy(model)
    by_id = {b["id"]:b for b in child["blocks"]}
    history = child.get("provenance", {}).get("merge_history", [])
    if not history:
        raise ValueError("Only an initialized merge trial has free parameters")
    merged_id = history[-1]["new_block_id"]
    seen = set()
    for spec, coordinate in zip(free_specs, coordinates):
        block = by_id.get(spec["block_id"])
        if block is None or spec["block_id"] != merged_id:
            raise ValueError("Free parameters must belong to the declared merged block")
        field = spec["field"]
        key = (spec["block_id"], field, spec.get("index"))
        if key in seen:
            raise ValueError("Duplicate free parameter specification")
        seen.add(key)
        transform = spec.get("transform", "log")
        if field == "alpha_fraction":
            if block["kind"] != "massive" or transform != "logit":
                raise ValueError("Only massive merged-block allocations use the logit transform")
            index = spec.get("index")
            if not isinstance(index,int) or index not in range(4):
                raise ValueError("Unknown source index")
            budget = _positive(spec["budget"], "alpha budget")
            # Read immutable input allocation for validation, not a previous update.
            original_merged = next(b for b in model["blocks"] if b["id"]==merged_id)
            initial_budget = float(model["air"]["alpha"][index]+original_merged["alpha"][index])
            if not math.isclose(budget,initial_budget,rel_tol=1e-12,abs_tol=1e-14):
                raise ValueError("Alpha budget must equal the initialized merged-plus-air pool")
            small = math.exp(-abs(float(coordinate)))
            if coordinate >= 0:
                fraction,complement=1.0/(1.0+small),small/(1.0+small)
            else:
                fraction,complement=small/(1.0+small),1.0/(1.0+small)
            block["alpha"][index]=budget*fraction
            child["air"]["alpha"][index]=budget*complement
            continue
        if transform != "log":
            raise ValueError("Positive R/C parameters require the log transform")
        if abs(coordinate)>80:
            raise ValueError("R/C log coordinates exceed the numerical guard")
        value=math.exp(float(coordinate))
        if field in ("r_boundary", "r_direct"):
            required_kind="massive" if field=="r_boundary" else "massless"
            index=spec.get("index")
            if block["kind"]!=required_kind or not isinstance(index,int) or index not in range(4) or block[field][index] is None:
                raise ValueError("Free resistance refers to an inactive or incompatible port")
            block[field][index]=value
        elif field in ("r_air", "C") and block["kind"]=="massive":
            block[field]=value
        else:
            raise ValueError("Unsupported merged-block parameter")
    validate_model(child)
    return child
