"""Frozen context Gaussian proposals with exact state-dependent MH correction.

Mixture components/weights are fixed for the unchanged outside-block context.
Within a block, a component is selected with its Gaussian-mixture
responsibility at x. Each selected proposal is a centered Gaussian around x,
with covariance concentrating on the slow-interior time-constant direction.
The resulting proposal is state dependent, so its complete normalized mixture
density is evaluated in both directions. No distribution below is a prior.
"""
import numpy as np
from scipy.special import logsumexp

INTERIOR=np.array([8,9,10,11,16,21,22,24],dtype=np.int64)

def log_responsibilities(mix,x):
    standardized=np.einsum('kij,kj->ki',mix.inv_cholesky,np.asarray(x)-mix.means)
    terms=mix.const-.5*np.sum(standardized*standardized,axis=1)
    return terms-logsumexp(terms)

def direction_covariances(mix,index,scale=.3,nugget=.01):
    columns=mix.covariances[:,:,index]
    directions=columns/mix.covariances[:,index,index,None]
    cov=scale**2*np.einsum('ki,kj->kij',directions,directions)+nugget**2*mix.covariances
    return (cov+np.swapaxes(cov,1,2))/2

def centered_component_logpdf(delta,cholesky):
    standardized=np.einsum('kij,j->ki',np.linalg.inv(cholesky),delta)
    dimension=len(delta)
    return -.5*np.sum(standardized*standardized,axis=1)-np.log(np.diagonal(cholesky,axis1=1,axis2=2)).sum(1)-dimension/2*np.log(2*np.pi)

def directed_logq(mix,x,y,cholesky):
    return float(logsumexp(log_responsibilities(mix,x)+centered_component_logpdf(np.asarray(y)-x,cholesky)))

def directed_proposal(mix,x,rng,index,scale=.3,nugget=.01):
    x=np.asarray(x);logweights=log_responsibilities(mix,x)
    component=int(rng.choice(len(logweights),p=np.exp(logweights)))
    cholesky=np.linalg.cholesky(direction_covariances(mix,index,scale,nugget))
    delta=cholesky[component]@rng.normal(size=len(x));y=x+delta
    component_density=centered_component_logpdf(delta,cholesky)
    forward=logsumexp(logweights+component_density)
    reverse=logsumexp(log_responsibilities(mix,y)+component_density)
    return y,float(reverse-forward),dict(component=component,forward_logq=float(forward),reverse_logq=float(reverse),direction_scale=float(scale),nugget=float(nugget))

def conditional_curve_mean(mix,value,index):
    variance=mix.covariances[:,index,index]
    terms=np.log(mix.weights)-.5*(np.log(2*np.pi*variance)+(value-mix.means[:,index])**2/variance)
    logweights=terms-logsumexp(terms)
    means=mix.means+mix.covariances[:,:,index]*(value-mix.means[:,index])[:,None]/variance[:,None]
    mean=np.exp(logweights)@means;mean[index]=value
    return mean

def curve_map(mix,x,index,delta,eta):
    y=np.asarray(x).copy();value=float(x[index]);new=value+delta
    y+=conditional_curve_mean(mix,new,index)-conditional_curve_mean(mix,value,index)+eta;y[index]=new
    return y

def curve_proposal(mix,x,rng,index,scale=.3,nugget=.01):
    # Outside-block context is unchanged, so mix is the same at x and y.
    # The scalar increment and other-coordinate noise have symmetric laws.
    # The triangular mean-following shear has unit volume Jacobian.
    others=np.array([j for j in range(len(x)) if j!=index]);component=int(rng.choice(len(mix.weights),p=mix.weights))
    eta=np.zeros(len(x));eta[others]=nugget*np.linalg.cholesky(mix.covariances[component][np.ix_(others,others)])@rng.normal(size=len(others))
    delta=scale*rng.normal();y=curve_map(mix,x,index,delta,eta)
    return y,0.,dict(component=component,tau_increment=float(delta),direction_scale=float(scale),nugget=float(nugget))
