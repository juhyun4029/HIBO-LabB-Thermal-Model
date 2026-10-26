"""Numerical checks for retained transitions and the latest fixed window.

At most 1% of ordinary HMC transitions may be invalid or have absolute energy
error above 1000. Each chain must attempt and accept ordinary HMC moves.
These engineering thresholds assess numerical behavior separately from the
R-hat and ESS convergence criteria. Evaluation preserves draws, kernel state,
and sampling random streams."""
import numpy as np


RECENT_DRAWS = 4000
MAX_BAD_FRACTION = .01
STATISTIC_COLUMNS = 38


def _window_health(stats, name, first_draw):
    blocking, chains = [], []
    for chain, rows in enumerate(stats, start=1):
        # kind=0 identifies ordinary HMC. Bridge/conditional-heat proposals
        # are distinct transition kernels and must not dilute its denominator.
        if not np.isfinite(rows[:, 0]).all():
            blocking.append(f'{name}, chain {chain}: nonfinite transition-kind flags')
        hrows = rows[rows[:, 0] == 0]
        count = len(hrows)
        entry = dict(chain=chain, hmc_attempts=int(count), hmc_accepted=0,
                     numerically_invalid_count=0, abs_energy_over_1000_count=0,
                     invalid_or_abs_energy_over_1000_count=0,
                     hmc_acceptance=None, numerically_invalid_fraction=None,
                     abs_energy_over_1000_fraction=None,
                     invalid_or_abs_energy_over_1000_fraction=None,
                     length_capped_count=0, length_capped_fraction=None,
                     finite_health_flags=True)
        if not count:
            blocking.append(f'{name}, chain {chain}: no ordinary HMC attempts')
        else:
            flags = hrows[:, [1, 5, 15, 28, 29]]
            finite = bool(np.isfinite(flags).all())
            entry['finite_health_flags'] = finite
            if not finite:
                blocking.append(f'{name}, chain {chain}: nonfinite HMC health flags')
            elif not np.isin(flags, [0., 1.]).all():
                blocking.append(f'{name}, chain {chain}: HMC health flags are not binary')
            # Counts remain JSON-safe even if a damaged checkpoint has NaNs;
            # the finite-flags failure above then blocks completion explicitly.
            accepted, bad, capped, invalid, large = (
                int(np.count_nonzero(np.isfinite(flags[:, j]) & (flags[:, j] == 1.)))
                for j in range(5))
            entry.update(hmc_accepted=accepted,
                         numerically_invalid_count=invalid,
                         abs_energy_over_1000_count=large,
                         invalid_or_abs_energy_over_1000_count=bad,
                         hmc_acceptance=accepted/count,
                         numerically_invalid_fraction=invalid/count,
                         abs_energy_over_1000_fraction=large/count,
                         invalid_or_abs_energy_over_1000_fraction=bad/count,
                         length_capped_count=capped,
                         length_capped_fraction=capped/count)
            if accepted < 1:
                blocking.append(f'{name}, chain {chain}: no accepted ordinary HMC transition')
            if bad/count > MAX_BAD_FRACTION:
                blocking.append(f'{name}, chain {chain}: more than 1% invalid or |energy error|>1000 HMC transitions ({bad}/{count})')
        chains.append(entry)
    return dict(passed=not blocking, draws_per_chain=int(stats.shape[1]),
                first_retained_draw=int(first_draw),
                last_retained_draw=int(first_draw+stats.shape[1]-1),
                chain_health=chains, blocking_failures=blocking)


def production_health(stats):
    """Check a ``(chain, retained_draw, 38)`` statistics array without mutation.

    Both the complete retained history and the last min(4000, draws) must
    pass independently. This prevents an early bad period from disappearing
    from the audit or a recently bad period from being diluted by long history.
    Missing/malformed/empty statistics fail closed with an explicit reason.
    """
    result = dict(schema='hibo-production-health-v1',
                  stage='retained_production', passed=False,
                  blocking_failures=[], all_retained=None, recent=None,
                  gate=dict(max_invalid_or_abs_energy_over_1000_fraction=MAX_BAD_FRACTION,
                            abs_energy_error_threshold=1000,
                            recent_draws_per_chain=RECENT_DRAWS,
                            at_least_one_hmc_attempt_per_chain=True,
                            at_least_one_accepted_hmc_per_chain=True,
                            finite_health_flags=True),
                  threshold_basis='Inherited settling engineering guard; 1% and |energy error|>1000 are not universal literature defaults or posterior precision criteria.',
                  policy='Check all retained draws and the recent fixed window; retain every draw; no adaptation, thinning, sample deletion, or RNG use.',
                  rhat_and_ess_are_separate_required_gates=True)
    try:
        statistics = np.asarray(stats, dtype=float)
    except (TypeError, ValueError):
        result['blocking_failures'] = ['Retained HMC statistics are missing or not numeric']
        return result
    if statistics.ndim != 3 or statistics.shape[2] != STATISTIC_COLUMNS:
        result['blocking_failures'] = ['Retained HMC statistics require (chain, draw, 38) axes']
        return result
    if statistics.shape[0] == 0 or statistics.shape[1] == 0:
        result['blocking_failures'] = ['No retained chains/draws are available for numerical health checks']
        return result
    count = statistics.shape[1]
    recent_count = min(RECENT_DRAWS, count)
    result['all_retained'] = _window_health(statistics, 'all retained', 1)
    result['recent'] = _window_health(statistics[:, -recent_count:], 'recent window', count-recent_count+1)
    result['blocking_failures'] = (result['all_retained']['blocking_failures']
                                   + result['recent']['blocking_failures'])
    result['passed'] = not result['blocking_failures']
    return result
