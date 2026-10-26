import numpy as np
from numba import njit
from profile_coordinates import profile_gradient
from reflected_hmc_integrator import reflected_drift,PAIRS

@njit(cache=True,nogil=True)
def profile_trajectory(q0,p0,step,nsteps,inverse,args):
    q=q0.copy();p=p0.copy();value,g=profile_gradient(q,*args);p+=.5*step*g
    for k in range(nsteps):
        q,p,collisions,valid=reflected_drift(q,p,inverse,step,PAIRS)
        if not valid:return q,p,-np.inf,False
        value,g=profile_gradient(q,*args)
        if not np.isfinite(value) or not np.isfinite(g).all():return q,p,-np.inf,False
        p+=(.5 if k==nsteps-1 else 1.)*step*g
    return q,-p,value,True
