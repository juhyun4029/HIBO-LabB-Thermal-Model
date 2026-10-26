"""Marginalize component labels sharing the same boundary condition.

The three exterior-wall and two interior-wall likelihoods are exchangeable.
Their distinct IDF-based priors are summed over label permutations. Ordered
time constants remove the 3! * 2! equivalent representations. Each record
restores labels using their conditional prior probabilities."""
import itertools
import numpy as np
from scipy.special import logsumexp
from model import Model as PhysicalModel
from physical_prior import physical_prior_gradient

PERMS=[]
for a in itertools.permutations([1,2,3]):
 for b in itertools.permutations([4,5]):
  p=np.arange(55)
  for dst,src in zip([1,2,3,4,5],[*a,*b]):
   for i,j in [(2*dst,2*src),(2*dst+1,2*src+1),(17+dst,17+src),*[(27+7*k+dst,27+7*k+src) for k in range(4)]]:p[i]=j
  PERMS.append(p)
PERMS=np.array(PERMS)

def canonicalize(u):
 u=np.array(u,copy=True);shape=u.shape;flat=u.reshape(-1,55)
 for group in [[1,2,3],[4,5]]:
  ordered=np.argsort(flat[:,np.array(group)+17],axis=1);original=flat.copy()
  for j,dst in enumerate(group):
   src=np.array(group)[ordered[:,j]];rows=np.arange(len(flat))
   for target,source in [(2*dst,2*src),(2*dst+1,2*src+1),(17+dst,17+src),*[(27+7*k+dst,27+7*k+src) for k in range(4)]]:flat[:,target]=original[rows,source]
 return flat.reshape(shape)

def to_v(u):
 u=canonicalize(u);v=u.copy()
 for group in [[18,19,20],[21,22]]:
  for prev,cur in zip(group[:-1],group[1:]):v[...,cur]=np.log(np.maximum(u[...,cur]-u[...,prev],1e-14))
 return v

def to_u(v):
 u=np.array(v,copy=True)
 for group in [[18,19,20],[21,22]]:
  for prev,cur in zip(group[:-1],group[1:]):u[...,cur]=u[...,prev]+np.exp(v[...,cur])
 return u

class CanonicalModel:
 def __init__(self,period):self.base=PhysicalModel(period);self.period=period;self.ndim=55
 def mixture(self,u,gradient=False):
  ys=u[PERMS];pn=self.base.native(ys);priors=-.5*np.sum(((pn[:,:25]-self.base.mean[:25])/self.base.sd[:25])**2,axis=1);total=logsumexp(priors);w=np.exp(priors-total)
  if not gradient:return total,w,ys,priors
  gs=np.array([physical_prior_gradient(self.base,y)[np.argsort(p)] for y,p in zip(ys,PERMS)])
  return total,w,ys,priors,np.sum(w[:,None]*gs,axis=0)
 def logpost(self,v):
  if not np.isfinite(v).all() or np.max(v[[19,20,22]])>20:return -np.inf
  u=to_u(v);base=self.base.logpost(u)
  if not np.isfinite(base):return -np.inf
  total,_,_,pr=self.mixture(u);return base-pr[0]+total+np.sum(v[[19,20,22]])
 def gradient(self,v):
  if not np.isfinite(v).all() or np.max(v[[19,20,22]])>20:return -np.inf,np.zeros(55)
  u=to_u(v);lp,g=self.base.gradient(u)
  if not np.isfinite(lp):return -np.inf,np.zeros(55)
  total,_,_,pr,gmix=self.mixture(u,True);g=g-physical_prior_gradient(self.base,u)+gmix;gv=g.copy()
  for group in [[18,19,20],[21,22]]:
   gv[group[0]]=np.sum(g[group])
   for j in range(1,len(group)):gv[group[j]]=np.exp(v[group[j]])*np.sum(g[group[j:]])+1
  return lp-pr[0]+total+np.sum(v[[19,20,22]]),gv
 def draw_labels(self,v,rng,canonical_logpost=None):
  u=to_u(v);total,w,ys,priors=self.mixture(u);k=rng.choice(len(w),p=w)
  lp=self.logpost(v) if canonical_logpost is None else canonical_logpost
  return ys[k],float(lp-np.sum(v[[19,20,22]])-total+priors[k]),int(k)
