"""Heat-integrated multiple-try Metropolis update of the joint state.

A uniform anchor index and S-1 normalized proposal draws augment the current
thermal/heat state. A thermal proposal receives a fresh bank of S heat draws.
The acceptance ratio uses sums of importance weights, and a heat draw is
selected proportionally to its weight after acceptance or rejection. Removing
the auxiliary bank leaves the target joint thermal/heat posterior."""
import numpy as np
from scipy.special import gammaln,logsumexp
from conditional_heat import all_fractions
from marginal_heat import MarginalHeatModel
from global_transport import to_gap

def heat_logq(m,a,g):
    if hasattr(m,'logq_alpha'):return m.logq_alpha(a,g)+np.log(all_fractions(a)).sum()
    aa=all_fractions(a);eta=np.log(aa[:,:7]/aa[:,7,None]).ravel();z=g['iLeta']@(eta-g['eta']);r=z@z;D=28;df=m.df
    ln=-.5*(D*np.log(2*np.pi)+r)
    lt=gammaln((D+df)/2)-gammaln(df/2)-D/2*np.log(df*np.pi)-(D+df)/2*np.log1p(r/df)
    return float(np.logaddexp(np.log(m.normal_weight)+ln,np.log(1-m.normal_weight)+lt)-g['logdet_eta'])

def transition(m,q,newtheta,proposal_correction,rng,particles=256):
    theta=to_gap(q)[:25];noise=q[25:27];raw=m.to_raw(q) if hasattr(m,'to_raw') else __import__('effective_coordinates').to_raw(q);a=m.base.alpha(raw)[:,:7].ravel()
    oldlp,old=m.estimate(theta,noise,rng.normal(size=(particles,m.latent_dimension)))
    if old is None:raise FloatingPointError('Original multiple-try state invalid')
    anchor=int(rng.integers(particles));old['alphas'][anchor]=all_fractions(a)
    old['logweights'][anchor]=m.likelihood(a,old['g'])+np.log(all_fractions(a)).sum()-heat_logq(m,a,old['g'])
    oldlp=m.prior_a(theta,noise,m.mu0)+logsumexp(old['logweights'])-np.log(particles)
    newlp,new=m.estimate(newtheta,noise,rng.normal(size=(particles,m.latent_dimension)))
    if not np.isfinite(oldlp):
        raise FloatingPointError('Current multiple-try bank has nonfinite exact augmented density')
    delta=newlp-oldlp+proposal_correction
    valid_new=new is not None and np.isfinite(newlp) and np.isfinite(delta)
    prob=float(np.exp(min(0.,delta))) if valid_new else 0.0
    accepted=bool(np.log(rng.random())<delta) if valid_new else False
    bank=new if accepted else old
    weights=np.exp(bank['logweights']-logsumexp(bank['logweights']));j=int(rng.choice(particles,p=weights));heat=bank['alphas'][j,:,:7].ravel()
    qq=m.q_from(newtheta if accepted else theta,noise,heat)
    lp=m.model.logpost(qq)
    return qq,lp,dict(accepted=float(accepted),probability=prob,importance_ess=float(1/(weights@weights)),maximum_weight=float(weights.max()),selected_anchor=bool(not accepted and j==anchor))
