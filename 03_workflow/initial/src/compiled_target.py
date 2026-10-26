"""Compiled transformed-posterior density and gradient evaluation.

The algebra matches Model.gradient and CanonicalModel.gradient, which provide
Python reference implementations for comparison."""
import numpy as np
from numba import njit
from adjoint_fast import likelihood_gradient
from canonical_model import PERMS,CanonicalModel

@njit(cache=True)
def logistic(x):
    return np.exp(-np.logaddexp(0.,-x))

@njit(cache=True)
def native(u):
    p=u.copy()
    for i in range(7):
        t=u[2*i];r=u[2*i+1];lf=-np.logaddexp(0.,-r);lg=-np.logaddexp(0.,r)
        p[2*i]=t+lf;p[2*i+1]=t+lg;p[17+i]=u[17+i]-t-lf-lg
    p[14]=-u[14]+np.logaddexp(0.,-u[15]);p[15]=-u[14]+np.logaddexp(0.,u[15])
    p[25]=-5+7*logistic(u[25]);p[26]=-5+11*logistic(u[26])
    return p

@njit(cache=True)
def transform_gradient(u,g):
    gp=np.zeros(55);gp[:25]=g
    for i in range(7):
        f=logistic(u[2*i+1]);a=g[2*i];b=g[2*i+1];c=g[17+i]
        gp[2*i]=a+b-c;gp[2*i+1]=(1-f)*a-f*b+(2*f-1)*c;gp[17+i]=c
    f=logistic(u[15]);a=g[14];b=g[15];gp[14]=-a-b;gp[15]=-(1-f)*a+f*b
    return gp

@njit(cache=True)
def prior(u,mean):
    p=native(u);d=p[:25]-mean[:25];var=np.log(5.)
    return -.5*np.sum(d*d)/var,transform_gradient(u,-d/var)

@njit(cache=True,nogil=True)
def base_gradient(u,mean,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,capacity_divisor):
    p=native(u)
    if not np.isfinite(p).all() or np.max(np.abs(p[:25]))>80:return -np.inf,np.zeros(55)
    a=np.empty((4,8));aj=0.
    for k in range(4):
        z=np.zeros(8);z[:7]=u[27+7*k:34+7*k];mx=np.max(z);ex=np.exp(z-mx);a[k]=ex/np.sum(ex)
        aj+=np.sum(z)-8*(mx+np.log(np.sum(ex)))
    v=np.exp(p[:25]);v[17:25]/=capacity_divisor
    ls=p[25];rho=.98*logistic(p[26])
    ll,g,ga,gls,gr=likelihood_gradient(v,a,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,ls,rho)
    dif=p[:25]-mean[:25];pr=-.5*np.sum(dif*dif)/np.log(5.);g-=dif/np.log(5.)
    gp=transform_gradient(u,g)
    pn=-.5*((ls-np.log(.3))**2+((p[26]-1.5)/1.5)**2)
    jac=0.
    ng0=gls-(ls-np.log(.3));b=logistic(u[25]);gp[25]=ng0*7*b*(1-b)+1-2*b
    et=logistic(p[26]);ng1=gr*.98*et*(1-et)-(p[26]-1.5)/2.25;b=logistic(u[26]);gp[26]=ng1*11*b*(1-b)+1-2*b
    for i in range(2):jac+=np.log(7. if i==0 else 11.)-np.logaddexp(0.,-u[25+i])-np.logaddexp(0.,u[25+i])
    for k in range(4):
        ag=np.sum(ga[k]*a[k])
        for j in range(7):gp[27+7*k+j]=a[k,j]*(ga[k,j]-ag)+1-8*a[k,j]
    return ll+pr+aj+pn+jac,gp

@njit(cache=True,nogil=True)
def canonical_gradient(v,mean,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,capacity_divisor):
    if not np.isfinite(v).all() or max(v[19],v[20],v[22])>20:return -np.inf,np.zeros(55)
    u=v.copy();u[19]=u[18]+np.exp(v[19]);u[20]=u[19]+np.exp(v[20]);u[22]=u[21]+np.exp(v[22])
    val,g=base_gradient(u,mean,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,capacity_divisor)
    if not np.isfinite(val):return -np.inf,np.zeros(55)
    ps=np.empty(12);gs=np.empty((12,55))
    for k in range(12):
        perm=PERMS[k];pu=u[perm];ps[k],gg=prior(pu,mean)
        for j in range(55):gs[k,perm[j]]=gg[j]
    mx=np.max(ps);ew=np.exp(ps-mx);sw=np.sum(ew);weights=ew/sw;total=mx+np.log(sw)
    g-=gs[0]
    for k in range(12):g+=weights[k]*gs[k]
    out=g.copy()
    out[18]=g[18]+g[19]+g[20];out[19]=np.exp(v[19])*(g[19]+g[20])+1;out[20]=np.exp(v[20])*g[20]+1
    out[21]=g[21]+g[22];out[22]=np.exp(v[22])*g[22]+1
    return val-ps[0]+total+v[19]+v[20]+v[22],out

class CompiledCanonicalModel(CanonicalModel):
    def gradient(self,u):
        m=self.base;d=m.d
        return canonical_gradient(u,m.mean,m.bi,m.wi,d['boundaries'],d['g_hvac'],d['tsupply'],d['gains'],d['ginf'],getattr(m,'initial_temperature',float(d['y'][0])),d['y'],m.idx,m.contiguous,getattr(m,'capacity_divisor',1.))

@njit(cache=True,nogil=True)
def symmetric_gradient(u,mean,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,capacity_divisor):
    val,g=base_gradient(u,mean,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,capacity_divisor)
    if not np.isfinite(val):return -np.inf,np.zeros(55)
    ps=np.empty(12);gs=np.empty((12,55))
    for k in range(12):
        perm=PERMS[k];ps[k],gg=prior(u[perm],mean)
        for j in range(55):gs[k,perm[j]]=gg[j]
    mx=np.max(ps);ew=np.exp(ps-mx);sw=np.sum(ew);total=mx+np.log(sw)
    g-=gs[0]
    for k in range(12):g+=ew[k]/sw*gs[k]
    return val-ps[0]+total,g

class SymmetricModel(CanonicalModel):
    """Smooth symmetrized density without ordering or log-gap coordinates.

    f_sym(u) is proportional to sum_P f(Pu). Conditional restoration of a label
    with probability f(Pu)/sum_Q f(Qu) recovers the original labeled posterior.
    The omitted 1/12 factor is parameter independent.
    """
    def gradient(self,u):
        m=self.base;d=m.d
        return symmetric_gradient(u,m.mean,m.bi,m.wi,d['boundaries'],d['g_hvac'],d['tsupply'],d['gains'],d['ginf'],getattr(m,'initial_temperature',float(d['y'][0])),d['y'],m.idx,m.contiguous,getattr(m,'capacity_divisor',1.))
    def logpost(self,u):
        val=self.base.logpost(u)
        if not np.isfinite(val):return -np.inf
        total,_,_,pr=self.mixture(u)
        return val-pr[0]+total
    def draw_labels(self,u,rng,symmetric_logpost=None):
        total,w,ys,priors=self.mixture(u);k=rng.choice(len(w),p=w)
        val=self.logpost(u) if symmetric_logpost is None else symmetric_logpost
        return ys[k],float(val-total+priors[k]),int(k)

class OrderedRawModel(SymmetricModel):
    """Restrict raw log time constants to one ordering, without a gap Jacobian.

    The symmetrized density assigns equal mass to all twelve ordering domains.
    Exact conditional label restoration still recovers the physical posterior.
    """
    def logpost(self,u):
        if u[18]>u[19] or u[19]>u[20] or u[21]>u[22]:return -np.inf
        return super().logpost(u)
