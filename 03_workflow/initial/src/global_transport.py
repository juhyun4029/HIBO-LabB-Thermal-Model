"""A frozen Gaussian-mixture reversible mode-transport proposal.

Choose source k with q's conditional component probabilities at x. Choose
l with q's mixture weights. Preserve z=L_k^{-1}(x-mu_k), mapping to
mu_l+L_l z. This kernel is reversible with respect to q, including its affine
Jacobian. The MH correction for target pi is pi(y)q(x)/(pi(x)q(y)).
The map acts on unconstrained log-gap effective coordinates.
"""
import numpy as np
from scipy.special import logsumexp
GAPS=np.array([19,20,22])
def to_gap(q):
 v=np.array(q,copy=True)
 v[...,19]=np.log(q[...,19]-q[...,18]);v[...,20]=np.log(q[...,20]-q[...,19]);v[...,22]=np.log(q[...,22]-q[...,21]);return v

def from_gap(v):
 q=np.array(v,copy=True)
 q[...,19]=q[...,18]+np.exp(v[...,19]);q[...,20]=q[...,19]+np.exp(v[...,20]);q[...,22]=q[...,21]+np.exp(v[...,22]);return q

class MixtureTransport:
 def __init__(self,means,covariances,weights):
  self.means=np.asarray(means);self.covariances=np.asarray(covariances);self.weights=np.asarray(weights);self.cholesky=np.linalg.cholesky(covariances);self.inv_cholesky=np.linalg.inv(self.cholesky);self.const=np.log(weights)-np.log(np.diagonal(self.cholesky,axis1=1,axis2=2)).sum(-1)-.5*self.means.shape[-1]*np.log(2*np.pi)
 @classmethod
 def load(cls,path):
  z=np.load(path);return cls(z['means'],z['covariances'],z['weights'])
 def density(self,x):
  z=np.einsum('kij,kj->ki',self.inv_cholesky,x-self.means);terms=self.const-.5*(z*z).sum(-1);val=float(logsumexp(terms));return val,np.exp(terms-val)
 def propose(self,x,rng):
  q,prob=self.density(x);k=int(rng.choice(len(prob),p=prob));l=int(rng.choice(len(prob),p=self.weights));z=self.inv_cholesky[k]@(x-self.means[k]);y=self.means[l]+self.cholesky[l]@z;qp,_=self.density(y);return y,q-qp,k,l
