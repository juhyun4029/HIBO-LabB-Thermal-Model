"""Unbiased heat-fraction integration for exact pseudo-marginal MCMC.

Importance draws use a normalized normal/Student-t mixture in heat log-ratio
coordinates. The original Dirichlet(1,...,1) prior is constant in fractions,
so each importance weight includes the softmax Jacobian. An unbiased positive
estimate of the heat-integrated likelihood is retained in the extended state.
Base normals may be updated by a normal-reversible Crank-Nicolson move.
Selected heat fractions are drawn with the conditional importance weights;
this recovers the original joint posterior, rather than a Gaussian surrogate.
"""
import numpy as np
from scipy.special import gammaln,logsumexp,softmax,ndtri
from conditional_heat import ConditionalModel,all_fractions

class MarginalHeatModel(ConditionalModel):
    dimension=28
    df=8
    normal_weight=.8
    latent_dimension=37

    def estimate(self,theta,noise,latent,out=None,g=None):
        try:
            if out is None:out=self.basis(theta)
            if g is None:g=self.geometry(out,noise)
            z=latent[:,:28].copy();chi=(latent[:,28:36]**2).sum(1)
            use_t=latent[:,36]>ndtri(self.normal_weight)
            z[use_t]*=np.sqrt(self.df/chi[use_t,None])
            eta=g['eta'][None,:]+z@g['Leta'].T
            logits=np.concatenate([eta.reshape(-1,4,7),np.zeros((len(z),4,1))],axis=2)
            aa=softmax(logits,axis=2);a=aa[:,:,:7].reshape(-1,28)
            logaj=logits.sum((1,2))-8*logsumexp(logits,axis=2).sum(1)
            norm=(z*z).sum(1);D=self.dimension;df=self.df
            ln=-.5*(D*np.log(2*np.pi)+norm)
            lt=gammaln((D+df)/2)-gammaln(df/2)-D/2*np.log(df*np.pi)-(D+df)/2*np.log1p(norm/df)
            lq=np.logaddexp(np.log(self.normal_weight)+ln,np.log(1-self.normal_weight)+lt)-g['logdet_eta']
            B=g['Bw'];y=g['yw'];H=B.T@B;h=B.T@y
            squared=y@y-2*(a@h)+np.sum((a@H)*a,axis=1)
            if np.any(squared<-1e-5):raise FloatingPointError('Numerically negative quadratic residual')
            ll=-.5*squared-len(y)*g['ls']+.5*self.base.nblocks*np.log1p(-g['rho']**2)
            logweights=ll+logaj-lq
            logZ=float(logsumexp(logweights)-np.log(len(z)))
            prior=self.prior_a(theta,noise,self.mu0)
            lp=float(prior+logZ)
            if not np.isfinite(lp):return -np.inf,None
            return lp,dict(theta=theta.copy(),noise=noise.copy(),latent=latent.copy(),out=out,g=g,lp=lp,logZ=logZ,logweights=logweights,alphas=aa,conditional_ess=float(np.exp(2*logsumexp(logweights)-logsumexp(2*logweights))),max_weight=float(np.exp(logweights.max()-logsumexp(logweights))))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):return -np.inf,None

    def select_record(self,state,rng):
        weights=np.exp(state['logweights']-logsumexp(state['logweights']));j=int(rng.choice(len(weights),p=weights));a=state['alphas'][j,:,:7].ravel()
        q=self.q_from(state['theta'],state['noise'],a)
        original_lp=self.model.logpost(q)
        return self.model.draw_labels(q,rng,original_lp),q,j
