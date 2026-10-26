"""Independent optimization and dispersed chain initialization for July."""
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from canonical_model import canonicalize
from profile_coordinates import from_raw, to_raw
from tuned_adaptation import regularized_covariance


def fresh_positions(fine, seed, maxiter=1500, smoke=False):
    base=fine.base
    initial=base.initial()
    print(json.dumps(dict(stage='fresh_initialization_compile',old_draws_imported=0)),flush=True)
    base.gradient(initial)  # Compile once before independent CPU workers.
    starts=[]
    for c in range(4):
        rng=np.random.default_rng([seed,913,c])
        x=initial.copy()
        x[:25]+=rng.normal(0,.20,25)
        x[25:27]+=rng.normal(0,.08,2)
        x[27:]+=rng.normal(0,.25,28)
        starts.append(x)
    def fit(c):
        print(json.dumps(dict(stage='fresh_initialization_map',chain=c+1,maxiter=maxiter)),flush=True)
        def objective(x):
            value,gradient=base.gradient(x)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                return 1e100,np.zeros(55)
            return -value,-gradient
        if smoke:
            return starts[c],dict(chain=c+1,optimized=False,iterations=0)
        initial_value, initial_gradient = objective(starts[c])
        result=minimize(objective,starts[c],jac=True,method='L-BFGS-B',
                        bounds=[(-60.,60.)]*25+[(-12.,12.)]*2+[(-25.,25.)]*28,
                        options=dict(maxiter=maxiter,maxls=80,maxcor=20,ftol=1e-12,gtol=1e-5))
        attempts=[dict(method='L-BFGS-B',success=bool(result.success),message=str(result.message),iterations=int(result.nit))]
        candidates=[starts[c],result.x]
        candidate=min(candidates,key=lambda x:objective(x)[0])
        improvement=initial_value-objective(candidate)[0]
        # Retry stalled relative-objective termination with independent dense BFGS.
        # Optimization endpoints initialize chains and are excluded from retained draws.
        if not result.success or improvement <= 1e-8*(1+abs(initial_value)):
            print(json.dumps(dict(stage='fresh_initialization_retry',chain=c+1,
                                 reason='Unfinished or stalled L-BFGS-B search; independent BFGS retry')),flush=True)
            retry=minimize(objective,candidate,jac=True,method='BFGS',
                           options=dict(maxiter=maxiter,gtol=1e-5))
            attempts.append(dict(method='BFGS',success=bool(retry.success),message=str(retry.message),iterations=int(retry.nit)))
            candidates.append(retry.x)
            candidate=min(candidates,key=lambda x:objective(x)[0])
        final_value, final_gradient=objective(candidate)
        improvement=float(initial_value-final_value)
        gradient_max=float(np.max(np.abs(final_gradient)))
        if (not np.isfinite(final_value) or not np.isfinite(final_gradient).all()
                or (improvement <= 1e-8*(1+abs(initial_value)) and gradient_max>1e-3)):
            raise RuntimeError(f'Fresh initialization for chain {c+1} stalled; no retained sampling was started.')
        metadata=dict(chain=c+1,optimized=True,
                              optimizer_success=bool(attempts[-1]['success']),attempts=attempts,
                              iterations=sum(attempt['iterations'] for attempt in attempts),
                              initial_logposterior=float(-initial_value),logposterior=float(-final_value),
                              objective_improvement=improvement,gradient_max_abs=gradient_max,
                              optimizer_success_is_not_an_equilibration_claim=True)
        print(json.dumps(dict(stage='fresh_initialization_chain_complete',**metadata)),flush=True)
        return candidate,metadata
    with ThreadPoolExecutor(4) as pool:
        results=list(pool.map(fit,range(4)))
    raw=canonicalize(np.array([value[0] for value in results]))
    # Strictly separate exact construction-time-constant ties. This is only
    # initialization; every quantity is subsequently free in the exact kernel.
    for c in range(4):
        for group in [[18,19,20],[21,22]]:
            for previous,current in zip(group[:-1],group[1:]):
                raw[c,current]=max(raw[c,current],raw[c,previous]+1e-4)
    q=from_raw(raw)
    if not np.isfinite([fine.logpost(x) for x in q]).all():
        raise FloatingPointError('Fresh July starting points must have finite density.')
    covariance=np.eye(55)*.25
    if not smoke:
        anchor=q[int(np.argmax([fine.logpost(x) for x in q]))]
        hessian=-approx_derivative(lambda x:fine.gradient(x)[1],anchor,method='3-point')
        hessian=(hessian+hessian.T)/2
        scale=1/np.sqrt(np.maximum(np.abs(np.diag(hessian)),.25))
        normalized=hessian*np.outer(scale,scale)
        eig,vec=np.linalg.eigh((normalized+normalized.T)/2)
        covariance=((vec*(1/np.clip(eig,.05,1e8)))@vec.T)*np.outer(scale,scale)
        covariance=regularized_covariance(covariance,.20)
        # Disperse independently around each fresh local search endpoint.
        # This is an initialization device, not a bound on the sampled target.
        factor=np.linalg.cholesky(covariance)
        for c in range(4):
            rng=np.random.default_rng([seed,914,c])
            local_logposterior=float(fine.logpost(q[c]))
            for attempt in range(32):
                # A local Hessian is a poor global approximation. Backtrack
                # dispersion rather than accepting an extremely remote point
                # merely because its density happens to remain finite.
                dispersion=.5*.5**(attempt//4)
                proposal=q[c]+dispersion*(factor@rng.normal(size=55))
                proposal=from_raw(canonicalize(to_raw(proposal)))
                value=float(fine.logpost(proposal))
                if np.isfinite(value) and value>=local_logposterior-.5*len(proposal):
                    q[c]=proposal
                    results[c][1].update(initial_dispersion_scale=dispersion,
                        initial_dispersion_attempts=attempt+1,
                        initial_dispersion_backtracks=attempt//4,
                        initial_dispersion_baseline_logposterior=local_logposterior,
                        dispersed_logposterior=value,
                        local_dispersion_logdensity_loss_limit=.5*len(proposal))
                    break
            else:
                raise RuntimeError(f'Could not construct an independent finite dispersed start for chain {c+1}.')
    return q,covariance,dict(initialization='Four independent seeded July searches with stalled-search fallback and independent Hessian-scaled dispersion; all discarded',
                              source_checkpoint=None,source_retained_per_chain=0,
                              old_production_pooled=False,source_coordinate_conversion=False,
                              optimizer=results and [r[1] for r in results],
                              maximum_independent_dispersion_scale=.5,
                              dispersion_local_logdensity_loss_limit=27.5,
                              dispersion_guard_applies_only_to_initialization=True,
                              smoke_test=bool(smoke))


def initial_mixture(q,covariance):
    """Broad auxiliary proposal; fitted warmup replaces it before production."""
    from global_transport import MixtureTransport,to_gap
    samples=to_gap(q)[:,:25]
    means=samples.copy()
    # Initial only; conditional MH corrections keep the target exact.
    cov=np.eye(25)*.20
    return MixtureTransport(means,np.repeat(cov[None,:,:],4,axis=0),np.full(4,.25))
