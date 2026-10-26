"""Frozen Metropolis transport for the exterior/air-capacity partition.

The proposal is fitted to discarded warmup draws from the current run. It
jointly updates the fastest exterior log(tau), log(E/C_air), and log(C_air).
Other thermal coordinates and noise parameters remain fixed during this move;
ordinary HMC updates all 55 free coordinates. Heat allocations use the exact
multiple-try kernel with its full importance-weight correction.

The log(tau) < 5 and >= 5 regions define proposal components, not prior
constraints or posterior weights. Metropolis correction accounts for the
balanced proposal. Fitting ends before fixed-kernel settling and production;
resume restores the frozen proposal arrays."""
import numpy as np
from scipy.special import logsumexp

from global_transport import MixtureTransport, to_gap, GAPS
from multiple_try_heat import transition as heat_transition
from tuned_adaptation import regularized_covariance

PARTITIONS = np.array([1, 3, 9, 13], dtype=int)
REGION_SPLIT = 5.0


def coordinates(q):
    """Extract a unit-Jacobian replacement of (q18, q3, q24)."""
    q = np.asarray(q, dtype=float)
    parts = np.concatenate([np.zeros((*q.shape[:-1], 1)), q[..., PARTITIONS]], axis=-1)
    log_air = q[..., 24] - logsumexp(parts, axis=-1)
    return np.stack([q[..., 18], q[..., 3], log_air], axis=-1)


def embed(q, values):
    """Inverse coordinate map; preserve every other profile coordinate."""
    answer = np.array(q, copy=True, dtype=float)
    answer[18], answer[3] = values[0], values[1]
    answer[24] = values[2] + logsumexp(np.r_[0.0, answer[PARTITIONS]])
    return answer


def fit(records, window=4000, min_count=64, shrinkage=0.20):
    """Fit two proposal components from discarded warmup draws.
    
    Return (None, metadata) if either region has insufficient observations. The
    auxiliary proposal is then disabled while the target and convergence criteria
    remain unchanged."""
    max_iterations, min_region_draws = window, min_count
    metadata = dict(enabled=False, source='this_run_discarded_warmup_only',
                    coordinates=['fastest_exterior_log_tau',
                                 'fastest_exterior_log_effective_capacity_over_air',
                                 'log_air_capacity'],
                    region_split=REGION_SPLIT, minimum_region_draws=int(min_region_draws),
                    maximum_warmup_iterations=int(max_iterations),
                    covariance_shrinkage=float(shrinkage),
                    proposal_weights=[0.5, 0.5], target_changed=False,
                    old_run_samples_used=False)
    if max_iterations < 1 or min_region_draws < 4:
        raise ValueError('Invalid transport fitting limits')
    array = np.asarray(records, dtype=float)
    if array.size == 0:
        metadata['reason'] = 'no_discarded_warmup_positions'
        return None, metadata
    if array.ndim not in (2, 3) or array.shape[-1] != 55:
        raise ValueError('Transport fitting expects (iterations[, chains], 55)')
    recent = array[-max_iterations:]
    metadata.update(warmup_iterations_used=int(len(recent)),
                    total_positions_used=int(recent.size // 55))
    if not np.isfinite(recent).all():
        metadata['reason'] = 'nonfinite_discarded_warmup_positions'
        return None, metadata
    values = coordinates(recent.reshape(-1, 55))
    fast = values[:, 0] < REGION_SPLIT
    groups = [values[~fast], values[fast]]
    metadata['region_counts'] = [int(len(group)) for group in groups]
    if min(map(len, groups)) < min_region_draws:
        metadata['reason'] = 'both_proposal_regions_require_enough_warmup_positions'
        return None, metadata
    try:
        means = np.asarray([group.mean(axis=0) for group in groups])
        covariances = np.asarray([
            regularized_covariance(np.cov(group, rowvar=False), shrinkage,
                                   eigen_floor=0.001)
            for group in groups])
        mixture = _validated_mixture(means, covariances, np.array([0.5, 0.5]))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        metadata.update(reason='proposal_fit_unavailable', detail=str(error))
        return None, metadata
    metadata.update(enabled=True, reason='frozen_two_region_proposal_fitted',
                    means=means.tolist(), covariances=covariances.tolist())
    return mixture, metadata


def _validated_mixture(means, covariances, weights):
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (means.shape != (2, 3) or covariances.shape != (2, 3, 3)
            or weights.shape != (2,) or not np.isfinite(means).all()
            or not np.isfinite(covariances).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0,
                                                     rtol=0, atol=1e-12)):
        raise ValueError('Invalid frozen exterior transport arrays')
    if not np.allclose(covariances, covariances.transpose(0, 2, 1),
                       rtol=1e-12, atol=1e-14):
        raise ValueError('Transport covariance is not symmetric')
    np.linalg.cholesky(covariances)
    return MixtureTransport(means, covariances, weights)


def to_arrays(mixture, prefix='transport_'):
    """Safe numeric-only checkpoint representation, including disabled state."""
    if mixture is None:
        return {prefix+'means': np.empty((0, 3)),
                prefix+'covariances': np.empty((0, 3, 3)),
                prefix+'weights': np.empty(0)}
    return {prefix+'means': mixture.means.copy(),
            prefix+'covariances': mixture.covariances.copy(),
            prefix+'weights': mixture.weights.copy()}


def from_arrays(arrays, prefix='transport_'):
    means = arrays[prefix+'means']
    covariances = arrays[prefix+'covariances']
    weights = arrays[prefix+'weights']
    if (means.shape == (0, 3) and covariances.shape == (0, 3, 3)
            and weights.shape == (0,)):
        return None
    return _validated_mixture(means, covariances, weights)


def transition(model, q, logp, rng, mixture, particles=512):
    """One exact fixed-proposal move, with explicit identity/rejection counts.

    MixtureTransport's proposal ratio is in the unit-volume three-coordinate
    chart above. The heat kernel uses log-gap thermal coordinates and its
    target already includes the gap Jacobian. Consequently its proposal
    correction must additionally contain log(J_old)-log(J_new). No extra
    affine determinant is added: mixture reversibility already includes it.

    An invalid ordered proposal is a single rejection; it is not redrawn,
    sorted or clipped. Current-state failures in heat_transition propagate.
    """
    info = dict(available=mixture is not None, attempted=0, identity=0,
                cross_component=0, valid=0, accepted=0.0, probability=0.0,
                crossed_region=0, cross_region_proposal=0, region_crossing=0,
                importance_ess=0.0, maximum_weight=0.0,
                source_component=-1, destination_component=-1)
    if mixture is None:
        return q, logp, info
    info['attempted'] = 1
    proposed, correction, source, destination = mixture.propose(coordinates(q), rng)
    info.update(source_component=int(source), destination_component=int(destination),
                cross_component=int(source != destination))
    if source == destination:
        info.update(identity=1, valid=1)
        return q, logp, info
    candidate = embed(q, proposed)
    info['cross_region_proposal'] = int((candidate[18] < REGION_SPLIT) != (q[18] < REGION_SPLIT))
    if (not np.isfinite(candidate).all() or not np.isfinite(correction)
            or not candidate[18] < candidate[19] < candidate[20]
            or not candidate[21] < candidate[22]):
        return q, logp, info
    old_theta = to_gap(q)[:25]
    new_theta = to_gap(candidate)[:25]
    correction += float(old_theta[GAPS].sum() - new_theta[GAPS].sum())
    if not np.isfinite(new_theta).all() or not np.isfinite(correction):
        return q, logp, info
    answer, value, heat_info = heat_transition(model, q, new_theta,
                                               correction, rng, particles)
    if (not np.isfinite(answer).all() or not np.isfinite(value)
            or not np.isfinite(heat_info['probability'])):
        raise FloatingPointError('Nonfinite exact exterior transport endpoint')
    info.update(valid=1, accepted=float(heat_info['accepted']),
                probability=float(heat_info['probability']),
                crossed_region=int((answer[18] < REGION_SPLIT) != (q[18] < REGION_SPLIT)),
                region_crossing=int((answer[18] < REGION_SPLIT) != (q[18] < REGION_SPLIT)),
                importance_ess=float(heat_info['importance_ess']),
                maximum_weight=float(heat_info['maximum_weight']))
    return answer, value, info
