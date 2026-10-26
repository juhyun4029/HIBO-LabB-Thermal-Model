"""Gaussian-mixture proposals conditional on unchanged coordinate blocks."""
import numpy as np
from scipy.special import logsumexp
from global_transport import MixtureTransport

def conditional_mixture(mix,indices,x):
    b=np.asarray(indices);c=np.array([i for i in range(len(x)) if i not in b]);means=[];covs=[];logweights=[]
    for mu,C,w in zip(mix.means,mix.covariances,mix.weights):
        Ccc=C[np.ix_(c,c)];Cbc=C[np.ix_(b,c)];delta=x[c]-mu[c]
        L=np.linalg.cholesky(Ccc);reg=np.linalg.solve(Ccc,Cbc.T).T
        means.append(mu[b]+reg@delta);cond=C[np.ix_(b,b)]-reg@Cbc.T
        covs.append((cond+cond.T)/2)
        z=np.linalg.solve(L,delta)
        logweights.append(np.log(w)-np.log(np.diag(L)).sum()-.5*(z@z))
    logweights=np.asarray(logweights);weights=np.exp(logweights-logsumexp(logweights))
    return MixtureTransport(means,covs,weights)
