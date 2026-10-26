"""Reversible momentum-tempered HMC with exact cold-target correction.

Every thermal, allocation and noise coordinate participates in the same
55-dimensional trajectory. Only one kinetic direction, associated with the
fastest ordered exterior log time constant, is gradually heated then cooled.
Its dense-metric velocity direction jointly moves correlated R/C and gain
allocation coordinates. The density, priors, likelihood and ordered domain
are unchanged. No hot sample is retained.

For nsteps=2m and alpha=peak_temperature**(1/(2m)), each of the first m
steps is S_sqrt(alpha) L S_sqrt(alpha), and each remaining step uses inverse
scaling. L is one FULL reflected leapfrog step for the cold profile target.
The complete scale Jacobian is one, and reversal interchanges heating and
cooling steps. The caller applies the ordinary cold-endpoint Hamiltonian MH
test. The final momentum flip makes the returned deterministic map an
involution, including dense-metric ordered-boundary reflections. The rank-one
scaling is the kinetic-metric projection
  S_s(p) = p + (s-1)*e_d*(inverse[d,:] @ p)/inverse[d,d].
It has determinant s, inverse S_(1/s), and scales only one orthogonal kinetic
component, avoiding the entropy cost of heating 55 unrelated directions.
"""
import numpy as np
from numba import njit
from profile_coordinates import profile_gradient
from reflected_hmc_integrator import reflected_drift, PAIRS


@njit(cache=True, nogil=True)
def tempered_profile_trajectory(q0, p0, eps, nsteps, inverse, args,
                                peak_temperature, direction=18):
    if nsteps < 2 or nsteps % 2 != 0:
        raise ValueError('Tempered trajectory needs a positive even step count')
    if not np.isfinite(peak_temperature) or peak_temperature < 1.:
        raise ValueError('Peak momentum temperature must be finite and >=1')
    if direction < 0 or direction >= len(p0) or inverse[direction, direction] <= 0:
        raise ValueError('Tempered kinetic direction must have positive metric norm')
    q = q0.copy()
    p = p0.copy()
    value, gradient = profile_gradient(q, *args)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        return q, p, -np.inf, False
    half = nsteps // 2
    # Each full step changes the free momentum scale by alpha; after half
    # the trajectory its free kinetic temperature would reach the peak.
    scale = np.exp(np.log(peak_temperature)/(2.*nsteps))
    for iteration in range(nsteps):
        multiplier = scale if iteration < half else 1./scale
        projection = np.dot(inverse[direction], p)/inverse[direction, direction]
        p[direction] += (multiplier-1.)*projection
        # Keep both half-kicks: scaling between successive steps prevents
        # the ordinary HMC optimization that merges adjacent half-kicks.
        p += .5*eps*gradient
        q, p, collisions, valid = reflected_drift(q, p, inverse, eps, PAIRS)
        if not valid or not np.isfinite(q).all() or not np.isfinite(p).all():
            return q, p, -np.inf, False
        value, gradient = profile_gradient(q, *args)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            return q, p, -np.inf, False
        p += .5*eps*gradient
        projection = np.dot(inverse[direction], p)/inverse[direction, direction]
        p[direction] += (multiplier-1.)*projection
    return q, -p, value, True
