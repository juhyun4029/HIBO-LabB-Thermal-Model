"""Pre-production numerical checks and exploratory warmup stability reporting.

Warmup diagnostics are descriptive: adapting draws are not retained posterior
samples, and passing this report never proves convergence or identifiability.
Only invalid numerical states block production; mobility/precision flags are
recorded as warnings rather than selecting or dropping production samples.
"""
import numpy as np
from diagnostic_parallel import diagnostic_definition, diagnostic_matrix


def warmup_health(records, statistics, position, position_logp, steps, metric,
                  *, final_window=1000, smoke=False):
    # records: (warmup draw, chain, ordered coordinate)
    positions = np.asarray(records, dtype=float)
    stats = np.asarray(statistics, dtype=float)
    position = np.asarray(position, dtype=float)
    position_logp = np.asarray(position_logp, dtype=float)
    steps = np.asarray(steps, dtype=float)
    metric = np.asarray(metric, dtype=float)
    blocking = []
    if not (np.isfinite(position).all() and np.isfinite(position_logp).all()):
        blocking.append("Nonfinite final position or target log density")
    if not (np.isfinite(steps).all() and (steps > 0).all()):
        blocking.append("HMC step sizes must all be finite and positive")
    if not np.isfinite(metric).all():
        blocking.append("Nonfinite frozen HMC momentum metric")
    else:
        try:
            np.linalg.cholesky(metric)
        except np.linalg.LinAlgError:
            blocking.append("Frozen HMC momentum metric is not positive definite")
    count = min(int(final_window), len(positions))
    recent = positions[-count:] if count else positions
    recent_stats = stats[-count:] if count else stats
    warnings = []
    chains = []
    for chain in range(position.shape[0]):
        if len(recent_stats):
            rows = recent_stats[:, chain, :]
            hmc = rows[:, 0] == 0
            hrows = rows[hmc]
        else:
            hrows = np.empty((0, 18))
        entry = {"chain": chain + 1, "hmc_transitions": int(len(hrows))}
        if len(hrows):
            entry.update(hmc_acceptance=float(np.mean(hrows[:, 1])),
                         hmc_mean_acceptance_probability=float(np.mean(hrows[:, 2])),
                         invalid_or_abs_energy_over_1000_fraction=float(np.mean(hrows[:, 5])),
                         length_capped_fraction=float(np.mean(hrows[:, 15])),
                         hmc_leapfrog_mean=float(np.mean(hrows[:, 12])))
            if entry["hmc_acceptance"] == 0:
                warnings.append(f"Chain {chain+1}: no accepted HMC moves in final warmup window")
            if entry["invalid_or_abs_energy_over_1000_fraction"] > .01:
                warnings.append(f"Chain {chain+1}: more than 1% numerically invalid or very high energy HMC moves")
            if entry["length_capped_fraction"] > .5:
                warnings.append(f"Chain {chain+1}: more than half of HMC trajectories hit the leapfrog cap")
        chains.append(entry)
    exploratory = None
    if count >= 20 and np.isfinite(recent).all():
        # Deliberately ordered coordinates; this audit does not restore labels.
        v = np.swapaxes(recent, 0, 1)
        diag = diagnostic_matrix(v)
        valid = np.isfinite(diag).all()
        half = count // 2
        scale = np.std(v.reshape(-1, v.shape[-1]), axis=0, ddof=1)
        drift = np.abs(v[:, half:, :].mean(axis=1) - v[:, :half, :].mean(axis=1)) / np.maximum(scale, 1e-12)
        number = lambda x: float(x) if np.isfinite(x) else None
        exploratory = {"max_rhat": number(np.max(diag[0])), "min_bulk_ess": number(np.min(diag[1])),
                       "min_tail_ess": number(np.min(diag[2])),
                       "max_within_chain_half_mean_drift_over_pooled_sd": number(np.max(drift)),
                       "max_drift_per_chain": [number(x) for x in drift.max(axis=1)],
                       "valid": bool(valid), "used_for_production_convergence_gate": False}
        if not valid or np.max(diag[0]) >= 1.10:
            warnings.append("Final adapting warmup window has dispersed/undefined exploratory R-hat; production convergence is still unknown")
        if np.max(drift) > .5:
            warnings.append("Final warmup half-window means differ by more than 0.5 pooled SD in some coordinate")
    return {"schema": "hibo-warmup-health-v1", "stage": "discarded_adapting_warmup",
            "window_draws_per_chain": count, "total_discarded_warmup_per_chain": len(positions),
            "diagnostic_definition": diagnostic_definition(), "chain_health": chains,
            "exploratory_ordered_coordinate_stability": exploratory,
            "blocking_numerical_failures": blocking, "production_numerically_ready": not blocking,
            "warnings": warnings, "smoke": bool(smoke),
            "posterior_convergence_proven": False, "parameter_identifiability_proven": False,
            "policy": "All warmup discarded; no production thinning or posthoc removal; warnings do not alter retained draws"}


def evaluate_settling(fine, q_records, statistics, seed):
    """Gate a discarded FIXED-kernel settling window before production.

    ``fine`` is the sampler's ProfileModel, not only fine.base. Inputs are
    (chain, draw, 55) and (chain, draw, >=18). The restored-label RNG is private
    to this report; production position and all production RNGs are untouched.
    Every reported ESS remains descriptive and is not required to exceed 400.
    """
    from reporting import diagnostics, canonical_diagnostics
    from canonical_model import to_v
    from profile_coordinates import to_raw
    q_records = np.asarray(q_records, dtype=float)
    statistics = np.asarray(statistics, dtype=float)
    if q_records.ndim != 3 or q_records.shape[0] != 4 or q_records.shape[2] != 55:
        raise ValueError("Settling requires (4, draw, 55) ordered profile states")
    if (statistics.ndim != 3 or statistics.shape[:2] != q_records.shape[:2]
            or statistics.shape[2] < 18):
        raise ValueError("Settling statistic axes do not match states")
    if q_records.shape[1] < 500:
        raise ValueError("Settling gate needs at least 500 discarded fixed-kernel draws per chain")
    finite = bool(np.isfinite(q_records).all())
    if not finite:
        return {"schema": "hibo-fixed-settling-v1", "passed": False,
                "blocking_failures": ["Nonfinite discarded fixed-kernel states"],
                "retained_per_chain": 0, "discarded_window_draws_per_chain": q_records.shape[1],
                "diagnostic_definition": diagnostic_definition(),
                "production_convergence_proven": False}
    labeled = np.empty_like(q_records)
    for chain in range(4):
        label_rng = np.random.default_rng([int(seed), chain])
        for draw in range(q_records.shape[1]):
            # Label probabilities depend only on prior weights for each exact
            # likelihood-equivalent permutation. A supplied logp avoids a
            # redundant thermal solve; its returned numerical value is ignored.
            labeled[chain, draw], _, _ = fine.draw_labels(q_records[chain, draw], label_rng, 0.0)
    physical = diagnostics(fine.base, labeled)
    ordered = canonical_diagnostics(to_v(to_raw(q_records)))
    blocking = []
    for name, diag in (("physical", physical), ("ordered", ordered)):
        if diag["invalid_parameters"] or diag["max_rhat"] is None or diag["max_rhat"] >= 1.10:
            blocking.append(f"{name}: rank R-hat is undefined or >=1.10 in the discarded fixed-kernel window")
    chain_health, warnings = [], []
    for chain in range(4):
        hmc = statistics[chain, :, 0] == 0
        rows = statistics[chain, hmc]
        entry = {"chain": chain + 1, "hmc_transitions": int(len(rows))}
        if not len(rows):
            blocking.append(f"Chain {chain+1}: no HMC transitions in settling window")
        else:
            entry.update(hmc_acceptance=float(np.mean(rows[:,1])),
                         invalid_or_abs_energy_over_1000_fraction=float(np.mean(rows[:,5])),
                         length_capped_fraction=float(np.mean(rows[:,15])))
            if not np.isfinite(rows[:,[1,5,15]]).all():
                blocking.append(f"Chain {chain+1}: nonfinite HMC health statistics")
            if np.sum(rows[:,1]) == 0:
                blocking.append(f"Chain {chain+1}: no accepted HMC transition")
            if entry["invalid_or_abs_energy_over_1000_fraction"] > .01:
                blocking.append(f"Chain {chain+1}: more than 1% invalid or |energy error|>1000 HMC transitions")
            if entry["length_capped_fraction"] > .5:
                warnings.append(f"Chain {chain+1}: more than half of HMC trajectories hit the configured leapfrog cap")
        chain_health.append(entry)
    return {"schema": "hibo-fixed-settling-v1", "passed": not blocking,
            "stage": "discarded_fixed_kernel_settling", "retained_per_chain": 0,
            "discarded_window_draws_per_chain": int(q_records.shape[1]),
            "diagnostic_definition": diagnostic_definition(),
            "physical": physical, "ordered": ordered, "chain_health": chain_health,
            "blocking_failures": blocking, "warnings": warnings,
            "gate": {"rank_rhat_strictly_below": 1.10,
                     "bulk_and_tail_ess_are_descriptive_only": True,
                     "max_invalid_or_abs_energy_over_1000_fraction": .01,
                     "at_least_one_accepted_hmc_per_chain": True},
            "engineering_health_thresholds_are_not_posterior_precision_criteria": True,
            "production_convergence_proven": False,
            "production_rng_consumed": False, "old_production_imported": False}
