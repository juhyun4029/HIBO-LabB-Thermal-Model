"""Sequential truncated-Gaussian proposals on four allocation simplexes.

Weak Gaussian precision regularizes the proposal. Importance weights divide
out its normalized density, preserving the uniform Dirichlet priors and the
Gaussian likelihood evaluated along the minute-input trajectory."""
import numpy as np
from scipy.special import ndtr,logsumexp
from scipy.stats import truncnorm
from marginal_heat import MarginalHeatModel
from conditional_heat import all_fractions

class GHKHeatModel(MarginalHeatModel):
    def geometry(self,out,noise):
        g=super().geometry(out,noise)
        # The interior likelihood/Jacobian mode approximates the conditional
        # mean near an active positivity boundary. In FRACTION coordinates the
        # response remains linear, avoiding a curved log-ratio likelihood.
        mean=g['mode'];cov=1.3*(g['Lray']@g['Lray'].T)
        sd=np.sqrt(np.diag(cov));importance=np.minimum(ndtr(-mean/sd),ndtr((mean-1)/sd));order=np.argsort(-importance)
        # First handle entries with the strongest outside-simplex pressure.
        C=cov[np.ix_(order,order)];chol=np.linalg.cholesky((C+C.T)/2)
        g.update(ghk_mean=mean,ghk_order=order,ghk_chol=chol,ghk_scale=1.3)
        return g

    def draw_heat(self,g,latent):
        S=len(latent);a=np.zeros((S,28));remaining=np.ones((S,4));z=np.zeros((S,28));logq=np.zeros(S);L=g['ghk_chol'];order=g['ghk_order'];mean=g['ghk_mean'][order]
        for j,index in enumerate(order):
            source=index//7;conditional=mean[j]+z[:,:j]@L[j,:j];sd=L[j,j]
            lo=-conditional/sd;hi=(remaining[:,source]-conditional)/sd;u=ndtr(latent[:,j])
            zj=truncnorm.ppf(u,lo,hi);value=conditional+sd*zj
            if np.any(value<=0) or np.any(value>=remaining[:,source]) or not np.isfinite(value).all():raise ValueError('Numerically invalid sequential heat draw')
            a[:,index]=value;remaining[:,source]-=value;z[:,j]=zj
            logq+=truncnorm.logpdf(zj,lo,hi)-np.log(sd)
        aa=np.concatenate([a.reshape(S,4,7),remaining[:,:,None]],axis=2)
        return aa,logq

    def logq_alpha(self,a,g):
        a=np.asarray(a);remaining=np.ones(4);z=np.zeros(28);L=g['ghk_chol'];order=g['ghk_order'];mean=g['ghk_mean'][order];logq=0.
        for j,index in enumerate(order):
            source=index//7;conditional=mean[j]+z[:j]@L[j,:j];sd=L[j,j];lo=-conditional/sd;hi=(remaining[source]-conditional)/sd;z[j]=(a[index]-conditional)/sd
            logq+=truncnorm.logpdf(z[j],lo,hi)-np.log(sd);remaining[source]-=a[index]
        return float(logq)

    def estimate(self,theta,noise,latent,out=None,g=None):
        try:
            if out is None:out=self.basis(theta)
            if g is None:g=self.geometry(out,noise)
            aa,lq=self.draw_heat(g,latent);a=aa[:,:,:7].reshape(-1,28);B=g['Bw'];y=g['yw'];H=B.T@B;h=B.T@y
            squared=y@y-2*(a@h)+np.sum((a@H)*a,axis=1)
            if np.any(squared<-1e-5):raise FloatingPointError('Negative residual quadratic')
            ll=-.5*squared-len(y)*g['ls']+.5*self.base.nblocks*np.log1p(-g['rho']**2);lw=ll-lq
            logZ=float(logsumexp(lw)-np.log(len(latent)));lp=self.prior_a(theta,noise,self.mu0)+logZ
            if not np.isfinite(lp):return -np.inf,None
            return float(lp),dict(theta=theta.copy(),noise=noise.copy(),latent=latent.copy(),out=out,g=g,lp=float(lp),logZ=logZ,logweights=lw,alphas=aa,conditional_ess=float(np.exp(2*logsumexp(lw)-logsumexp(2*lw))),max_weight=float(np.exp(lw.max()-logsumexp(lw))))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):return -np.inf,None
