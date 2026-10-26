"""Volume-preserving coordinates for coupled thermal capacities and heat inputs.

This changes only sampling coordinates, not any physical prior or model.
For f_i=R1_i/(R1_i+R2_i), effective source log ratios are z_alpha+log(f_i).
The effective capacities C_i*f_i^2 of the fastest exterior/interior components
and the floor are partitioned with air C using a free total and log ratios.
Both transformations have unit absolute Jacobian in the existing log variables.
"""
import numpy as np
from numba import njit
from scipy.special import logsumexp
from compiled_target import OrderedRawModel,symmetric_gradient
GROUP=np.array([1,4,6],dtype=np.int64)

def from_raw(u):
 u=np.asarray(u);q=u.copy();lf=-np.logaddexp(0,-u[...,1:14:2]);q[...,27:]=(u[...,27:].reshape((*u.shape[:-1],4,7))+lf[...,None,:]).reshape((*u.shape[:-1],28));z=u[...,17+GROUP]+u[...,2*GROUP+1]-u[...,2*GROUP];q[...,24]=logsumexp(np.concatenate([u[...,24,None],z],axis=-1),axis=-1);q[...,2*GROUP+1]=z-u[...,24,None];return q

def to_raw(q):
 q=np.asarray(q);u=q.copy();parts=np.concatenate([np.zeros((*q.shape[:-1],1)),q[...,2*GROUP+1]],axis=-1);air=q[...,24]-logsumexp(parts,axis=-1);u[...,24]=air;u[...,2*GROUP+1]=air[...,None]+q[...,2*GROUP+1]+q[...,2*GROUP]-q[...,17+GROUP];lf=-np.logaddexp(0,-u[...,1:14:2]);u[...,27:]=(q[...,27:].reshape((*q.shape[:-1],4,7))-lf[...,None,:]).reshape((*q.shape[:-1],28));return u

@njit(cache=True)
def to_raw_compiled(q):
 u=q.copy();p=np.zeros(4)
 for j in range(3):p[j+1]=q[2*GROUP[j]+1]
 mx=np.max(p);logsum=mx+np.log(np.exp(p-mx).sum());air=q[24]-logsum;u[24]=air
 for j in range(3):
  i=GROUP[j];u[2*i+1]=air+q[2*i+1]+q[2*i]-q[17+i]
 for i in range(7):
  lf=-np.logaddexp(0.,-u[2*i+1])
  for k in range(4):u[27+7*k+i]=q[27+7*k+i]-lf
 return u

@njit(cache=True,nogil=True)
def effective_gradient(q,mean,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,capacity_divisor):
 u=to_raw_compiled(q);value,g=symmetric_gradient(u,mean,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,capacity_divisor)
 if not np.isfinite(value):return -np.inf,np.zeros(55)
 gr=np.zeros(7)
 for i in range(7):
  f=np.exp(-np.logaddexp(0.,-u[2*i+1]));ga=0.
  for k in range(4):ga+=g[27+7*k+i]
  gr[i]=g[2*i+1]-(1-f)*ga
 out=g.copy()
 for i in range(7):out[2*i+1]=gr[i]
 total=g[24]
 for i in GROUP:total+=gr[i]
 out[24]=total
 for i in GROUP:
  fraction=np.exp(u[17+i]+u[2*i+1]-u[2*i]-q[24]);out[2*i+1]=gr[i]-fraction*total;out[2*i]=g[2*i]+gr[i];out[17+i]=g[17+i]-gr[i]
 return value,out

class EffectiveModel(OrderedRawModel):
 def logpost(self,q):return super().logpost(to_raw(q))
 def gradient(self,q):
  m=self.base;d=m.d
  return effective_gradient(q,m.mean,m.bi,m.wi,d['boundaries'],d['g_hvac'],d['tsupply'],d['gains'],d['ginf'],getattr(m,'initial_temperature',float(d['y'][0])),d['y'],m.idx,m.contiguous,getattr(m,'capacity_divisor',1.))
 def draw_labels(self,q,rng,logpost=None):
  value=self.logpost(q) if logpost is None else logpost
  return super().draw_labels(to_raw(q),rng,value)
