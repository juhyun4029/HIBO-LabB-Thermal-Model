"""Likelihood-tempered reflected HMC, preserving the original cold posterior.

At beta, the target is prior(q) * likelihood(q)**beta. All prior factors and
coordinate Jacobians remain at power one. The auxiliary gradient only proposes
a reversible volume-preserving path; full minute-input endpoints correct it.
"""
import numpy as np
from numba import njit
from compiled_target import prior,PERMS,logistic
from effective_coordinates import to_raw_compiled,effective_gradient,GROUP
from reflected_hmc_integrator import reflected_drift,PAIRS

@njit(cache=True)
def transform_effective_gradient(q,u,g):
 gr=np.zeros(7)
 for i in range(7):
  f=logistic(u[2*i+1]);ga=0.
  for k in range(4):ga+=g[27+7*k+i]
  gr[i]=g[2*i+1]-(1-f)*ga
 out=g.copy()
 for i in range(7):out[2*i+1]=gr[i]
 total=g[24]
 for i in GROUP:total+=gr[i]
 out[24]=total
 for i in GROUP:
  fraction=np.exp(u[17+i]+u[2*i+1]-u[2*i]-q[24]);out[2*i+1]=gr[i]-fraction*total;out[2*i]=g[2*i]+gr[i];out[17+i]=g[17+i]-gr[i]
 return out

@njit(cache=True,nogil=True)
def effective_prior(q,mean):
 u=to_raw_compiled(q)
 if not np.isfinite(u).all():return -np.inf,np.zeros(55)
 ps=np.empty(12);gs=np.empty((12,55))
 for k in range(12):
  perm=PERMS[k];ps[k],g=prior(u[perm],mean)
  for j in range(55):gs[k,perm[j]]=g[j]
 mx=np.max(ps);ww=np.exp(ps-mx);sw=ww.sum();value=mx+np.log(sw);g=np.zeros(55)
 for k in range(12):g+=ww[k]/sw*gs[k]
 for k in range(4):
  z=np.zeros(8);z[:7]=u[27+7*k:34+7*k];mx=np.max(z);ex=np.exp(z-mx);a=ex/ex.sum()
  value+=np.sum(z)-8*(mx+np.log(ex.sum()))
  for j in range(7):g[27+7*k+j]=1-8*a[j]
 b=logistic(u[25]);ls=-5+7*b
 g[25]=-(ls-np.log(.3))*7*b*(1-b)+1-2*b
 b=logistic(u[26]);eta=-5+11*b
 g[26]=-(eta-1.5)/2.25*11*b*(1-b)+1-2*b
 value+=-.5*((ls-np.log(.3))**2+((eta-1.5)/1.5)**2)
 for i in range(2):value+=np.log(7. if i==0 else 11.)-np.logaddexp(0.,-u[25+i])-np.logaddexp(0.,u[25+i])
 return value,transform_effective_gradient(q,u,g)

@njit(cache=True,nogil=True)
def tempered_gradient(q,args,beta):
 pv,pg=effective_prior(q,args[0])
 if beta==0:return pv,pg
 fv,fg=effective_gradient(q,*args)
 return pv+beta*(fv-pv),pg+beta*(fg-pg)

@njit(cache=True,nogil=True)
def tempered_trajectory(position,momentum,step,nsteps,inverse_mass,args,beta):
 q=position.copy();p=momentum.copy();value,g=tempered_gradient(q,args,beta);p+=.5*step*g
 for k in range(nsteps):
  q,p,collisions,valid=reflected_drift(q,p,inverse_mass,step,PAIRS)
  if not valid:return q,p,-np.inf,False
  value,g=tempered_gradient(q,args,beta)
  if not np.isfinite(value) or not np.isfinite(g).all():return q,p,-np.inf,False
  p+=(.5 if k==nsteps-1 else 1.)*step*g
 return q,-p,value,True

def swap_log_ratio(beta_i,beta_j,loglike_i,loglike_j):
 return (beta_i-beta_j)*(loglike_j-loglike_i)

def draw_ordered_prior(model,rng):
 from scipy.stats import truncnorm
 from canonical_model import canonicalize
 from effective_coordinates import from_raw
 m=model.base;p=np.empty(55);p[:25]=m.mean[:25]+m.sd[:25]*rng.normal(size=25)
 p[25]=truncnorm.rvs((-5-np.log(.3)),(2-np.log(.3)),loc=np.log(.3),scale=1.,random_state=rng)
 p[26]=truncnorm.rvs((-5-1.5)/1.5,(6-1.5)/1.5,loc=1.5,scale=1.5,random_state=rng)
 a=rng.dirichlet(np.ones(8),size=4);p[27:]=np.log(a[:,:7]/a[:,7,None]).ravel()
 return from_raw(canonicalize(m.inverse(p)))
