"""Random-coordinate slice transition (Neal 2003, stepping out + shrinkage).

The width and maximum number of step-out intervals are fixed. Random allocation
of the finite step-out budget to left/right is essential for reversibility.
"""
import numpy as np

def coordinate_slice(x,logp,log_density,rng,indices,width=1.,max_steps=20):
    j=int(indices[rng.integers(len(indices))]);origin=float(x[j]);height=logp-rng.exponential();left=origin-width*rng.random();right=left+width;J=int(rng.integers(max_steps));K=max_steps-1-J;candidate=x.copy();evaluations=0
    def evaluate(value):
        nonlocal evaluations
        candidate[j]=value;evaluations+=1;return float(log_density(candidate))
    while J>0:
        if evaluate(left)<=height:break
        left-=width;J-=1
    while K>0:
        if evaluate(right)<=height:break
        right+=width;K-=1
    for _ in range(1000):
        value=rng.uniform(left,right);proposal_logp=evaluate(value)
        if np.isfinite(proposal_logp) and proposal_logp>=height:return candidate.copy(),proposal_logp,evaluations,j
        if value<origin:left=value
        else:right=value
    raise RuntimeError('Slice shrinkage did not terminate; calculation stopped without accepting a biased fallback')
