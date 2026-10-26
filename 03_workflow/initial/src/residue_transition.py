"""Augmented-label Metropolis pole/residue exchange.

Physical labels are sampled from their conditional prior probabilities.
Symmetric involutions swap (log tau, log E) between solid components, hold
their conductance totals fixed within the move, and exchange beta=f*alpha.
The simplex transformation includes its Jacobian correction. Canonicalization
follows the physical-space Metropolis substeps."""
import itertools
import numpy as np
from scipy.special import expit,softmax
from canonical_model import canonicalize
from profile_coordinates import from_raw

PAIRS=np.array(list(itertools.combinations(range(7),2)),dtype=int)

def exchange(u,i,j):
    v=u.copy();e=u[17:24]+u[1:14:2]-u[0:14:2]
    v[17+i],v[17+j]=u[17+j],u[17+i]
    v[2*i+1]=e[j]+u[2*i]-v[17+i]
    v[2*j+1]=e[i]+u[2*j]-v[17+j]
    old=softmax(np.c_[u[27:].reshape(4,7),np.zeros(4)],axis=1)
    new=old.copy();f=expit(u[1:14:2]);g=expit(v[1:14:2])
    if min(f[i],f[j],g[i],g[j])<=0:return None,None
    new[:,i]=old[:,j]*f[j]/g[i]
    new[:,j]=old[:,i]*f[i]/g[j]
    new[:,7]=1-new[:,:7].sum(1)
    if not np.all(new>0):return None,None
    v[27:]=np.log(new[:,:7]/new[:,7,None]).ravel()
    logjac=float(4*(np.log(f[i])+np.log(f[j])-np.log(g[i])-np.log(g[j]))+np.log(old).sum()-np.log(new).sum())
    return v,logjac

def transition(model,q,logp,rng,substeps=8):
    u,physical_lp,_=model.draw_labels(q,rng,logp)
    accepted=valid=nontrivial=0;ratios=[];pairs=[]
    for k in range(substeps):
        i,j=PAIRS[int(rng.integers(len(PAIRS)))];v,logjac=exchange(u,i,j)
        if v is None:continue
        valid+=1;new=model.base.logpost(v);delta=new-physical_lp+logjac
        if np.isfinite(new) and np.log(rng.random())<delta:
            nontrivial+=int(abs(u[17+i]-u[17+j])>.5)
            accepted+=1;u=v;physical_lp=new;pairs.append([int(i),int(j)])
        ratios.append(float(delta))
    if accepted:
        q=from_raw(canonicalize(u));logp=model.logpost(q)
    return q,logp,dict(attempted=substeps,valid=valid,accepted=accepted,nontrivial=nontrivial,pairs=pairs)
