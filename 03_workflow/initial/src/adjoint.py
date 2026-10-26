"""Exact discrete adjoint of the 17R8C backward-Euler star solve.

Input intervals are one minute and contain two 30-second substeps. The
likelihood interval is independent of this integration interval. Noise priors
refer to a one-hour innovation standard deviation and one-hour correlation.
"""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def likelihood_gradient(v, alpha, bi, wi, boundary, hvac_g, supply_t, gains,
                        inf_g, initial_t, observed, idx, contiguous,
                        log_sigma_hour, rho_hour, dt_seconds=60.,
                        likelihood_minutes=5.):
    n = len(observed)
    dt = dt_seconds / 2.
    cb = v[17:24] / dt
    ca = v[24] / dt
    gb = 1. / v[:14:2]
    gi = 1. / v[1:14:2]
    den = cb + gb + gi
    direct = np.zeros(4)
    for j in range(3):
        direct[wi[j]] += 1. / v[14+j]
    airbase = ca + np.sum(gi) + np.sum(direct)
    state = np.empty((2*n+1, 8))
    state[0, :] = initial_t
    for t in range(n):
        ad = airbase + hvac_g[t] + inf_g[t]
        inv = 1. / (ad - np.sum(gi*gi/den))
        for ss in range(2):
            k = 2*t+ss
            rhs = np.empty(7)
            rz = ca*state[k,7] + hvac_g[t]*supply_t[t] + inf_g[t]*boundary[t,1]
            for q in range(4):
                rz += direct[q]*boundary[t,q] + alpha[q,7]*gains[t,q]
            for i in range(7):
                rhs[i] = cb[i]*state[k,i] + gb[i]*boundary[t,bi[i]]
                for q in range(4):
                    rhs[i] += alpha[q,i]*gains[t,q]
                rz += gi[i]*rhs[i]/den[i]
            state[k+1,7] = rz*inv
            for i in range(7):
                state[k+1,i] = (rhs[i]+gi[i]*state[k+1,7])/den[i]
    if not np.isfinite(state).all():
        return -np.inf, np.zeros(25), np.zeros((4,8)), 0., 0.
    fraction = likelihood_minutes / 60.
    phi = rho_hour**fraction
    sigma2 = np.exp(2.*log_sigma_hour)*(1.-phi*phi)/(1.-rho_hour*rho_hour)
    e = state[2*(idx+1),7] - observed[idx]
    outgrad = np.zeros(n)
    quadratic = 0.
    first_n = 0
    dphi = 0.
    for j in range(len(idx)):
        if contiguous[j]:
            r = e[j]-phi*e[j-1]
            quadratic += r*r
            outgrad[idx[j]] -= r/sigma2
            outgrad[idx[j-1]] += phi*r/sigma2
            dphi += r*e[j-1]/sigma2
        else:
            quadratic += (1.-phi*phi)*e[j]*e[j]
            outgrad[idx[j]] -= (1.-phi*phi)*e[j]/sigma2
            dphi += phi*e[j]*e[j]/sigma2
            first_n += 1
    ll = -.5*quadratic/sigma2 - .5*len(idx)*np.log(sigma2) + .5*first_n*np.log1p(-phi*phi)
    gls = quadratic/sigma2 - len(idx)
    dphi -= first_n*phi/(1.-phi*phi)
    dpdr = fraction*phi/rho_hour
    dlsdr = -phi*dpdr/(1.-phi*phi) + rho_hour/(1.-rho_hour*rho_hour)
    grho = dphi*dpdr + gls*dlsdr
    grad = np.zeros(25)
    ga = np.zeros((4,8))
    lam = np.zeros(8)
    for k in range(2*n-1, -1, -1):
        t = k//2
        if k%2 == 1:
            lam[7] += outgrad[t]
        ad = airbase + hvac_g[t] + inf_g[t]
        inv = 1. / (ad - np.sum(gi*gi/den))
        qz = lam[7]
        for i in range(7):
            qz += gi[i]*lam[i]/den[i]
        qz *= inv
        q = np.empty(7)
        for i in range(7):
            q[i] = (lam[i]+gi[i]*qz)/den[i]
            grad[2*i] += q[i]*gb[i]*(state[k+1,i]-boundary[t,bi[i]])
            grad[2*i+1] += gi[i]*(q[i]-qz)*(state[k+1,i]-state[k+1,7])
            grad[17+i] += q[i]*cb[i]*(state[k,i]-state[k+1,i])
            for source in range(4):
                ga[source,i] += q[i]*gains[t,source]
        for j in range(3):
            grad[14+j] += qz/v[14+j]*(state[k+1,7]-boundary[t,wi[j]])
        grad[24] += qz*ca*(state[k,7]-state[k+1,7])
        for source in range(4):
            ga[source,7] += qz*gains[t,source]
        for i in range(7):
            lam[i] = cb[i]*q[i]
        lam[7] = ca*qz
    return ll, grad, ga, gls, grho
