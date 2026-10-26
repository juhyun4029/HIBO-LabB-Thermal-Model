"""Unit-Jacobian coordinates for conductance and effective capacitance.

Positive conductances are expressed as a free total and free log ratios.
The four fastest solid effective capacities share free partitions with air C.
All 17 resistances and eight capacities retain independent degrees of freedom."""
import numpy as np
from numba import njit
from scipy.special import logsumexp
from effective_coordinates import EffectiveModel,to_raw as old_to_raw,from_raw as old_from_raw
from compiled_target import symmetric_gradient,logistic
from tempered_hmc import effective_prior as old_prior

GROUP=np.array([0,1,4,6],dtype=np.int64)
EXT=np.array([2,4,6,14],dtype=np.int64)
INT=np.array([8,10,16],dtype=np.int64)

def from_raw(u):
    u=np.asarray(u);q=u.copy();lf=-np.logaddexp(0,-u[...,1:14:2]);q[...,27:]=(u[...,27:].reshape((*u.shape[:-1],4,7))+lf[...,None,:]).reshape((*u.shape[:-1],28))
    e=u[...,17+GROUP]+u[...,2*GROUP+1]-u[...,2*GROUP]
    q[...,24]=logsumexp(np.concatenate([u[...,24,None],e],axis=-1),axis=-1);q[...,2*GROUP+1]=e-u[...,24,None]
    for ids in [EXT,INT]:
        logs=np.concatenate([-u[...,ids[:-1]],u[...,ids[-1],None]],axis=-1)
        q[...,ids[0]]=logsumexp(logs,axis=-1);q[...,ids[1:]]=logs[...,:-1]-logs[...,-1,None]
    return q

def to_raw(q):
    q=np.asarray(q);u=q.copy()
    for ids in [EXT,INT]:
        z=np.concatenate([q[...,ids[1:]],np.zeros((*q.shape[:-1],1))],axis=-1);lg=q[...,ids[0],None]+z-logsumexp(z,axis=-1)[...,None]
        u[...,ids[:-1]]=-lg[...,:-1];u[...,ids[-1]]=lg[...,-1]
    parts=np.concatenate([np.zeros((*q.shape[:-1],1)),q[...,2*GROUP+1]],axis=-1);air=q[...,24]-logsumexp(parts,axis=-1);u[...,24]=air
    u[...,2*GROUP+1]=air[...,None]+q[...,2*GROUP+1]+u[...,2*GROUP]-q[...,17+GROUP]
    lf=-np.logaddexp(0,-u[...,1:14:2]);u[...,27:]=(q[...,27:].reshape((*q.shape[:-1],4,7))-lf[...,None,:]).reshape((*q.shape[:-1],28))
    return u

def to_old(q):return old_from_raw(to_raw(q))
def from_old(q):return from_raw(old_to_raw(q))

@njit(cache=True)
def to_raw_compiled(q):
    u=q.copy()
    for ids in (EXT,INT):
        z=np.zeros(len(ids))
        for j in range(len(ids)-1):z[j]=q[ids[j+1]]
        mx=np.max(z);s=mx+np.log(np.exp(z-mx).sum())
        for j in range(len(ids)-1):u[ids[j]]=-(q[ids[0]]+z[j]-s)
        u[ids[-1]]=q[ids[0]]-s
    parts=np.zeros(5)
    for j in range(4):parts[j+1]=q[2*GROUP[j]+1]
    mx=np.max(parts);ls=mx+np.log(np.exp(parts-mx).sum());air=q[24]-ls;u[24]=air
    for i in GROUP:u[2*i+1]=air+q[2*i+1]+u[2*i]-q[17+i]
    for i in range(7):
        lf=-np.logaddexp(0.,-u[2*i+1])
        for k in range(4):u[27+7*k+i]=q[27+7*k+i]-lf
    return u

@njit(cache=True,nogil=True)
def profile_gradient(q,mean,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,capacity_divisor):
    u=to_raw_compiled(q);value,g=symmetric_gradient(u,mean,bi,wi,boundaries,hvac,supply,gains,inf,initial,y,idx,contiguous,capacity_divisor)
    if not np.isfinite(value):return -np.inf,np.zeros(55)
    gr=np.zeros(7);out=g.copy()
    for i in range(7):
        ga=0.
        for k in range(4):ga+=g[27+7*k+i]
        gr[i]=g[2*i+1]-(1-logistic(u[2*i+1]))*ga;out[2*i+1]=gr[i]
    total=g[24]
    for i in GROUP:total+=gr[i]
    out[24]=total
    for i in GROUP:
        fraction=np.exp(u[17+i]+u[2*i+1]-u[2*i]-q[24]);out[2*i+1]=gr[i]-fraction*total;out[2*i]=g[2*i]+gr[i];out[17+i]=g[17+i]-gr[i]
    # Adjoint of each conductance-total/log-ratio transform.
    for ids in (EXT,INT):
        thermal=out[ids].copy();net=thermal[-1]
        for j in range(len(ids)-1):net-=thermal[j]
        out[ids[0]]=net
        for j in range(len(ids)-1):
            fraction=np.exp(-u[ids[j]]-q[ids[0]])
            out[ids[j+1]]=-thermal[j]-net*fraction
    return value,out

class ProfileModel(EffectiveModel):
    def logpost(self,q):return super().logpost(to_old(q))
    def gradient(self,q):
        m=self.base;d=m.d
        return profile_gradient(q,m.mean,m.bi,m.wi,d['boundaries'],d['g_hvac'],d['tsupply'],d['gains'],d['ginf'],getattr(m,'initial_temperature',float(d['y'][0])),d['y'],m.idx,m.contiguous,getattr(m,'capacity_divisor',1.))
    def draw_labels(self,q,rng,logpost=None):
        value=self.logpost(q) if logpost is None else logpost
        return super().draw_labels(to_old(q),rng,value)

def prior(q,mean):return old_prior(to_old(q),mean)[0]
