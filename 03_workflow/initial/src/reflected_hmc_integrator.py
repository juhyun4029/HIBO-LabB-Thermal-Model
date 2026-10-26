"""Leapfrog with exact kinetic reflections at ordered-time-constant boundaries.

A free drift reflects momentum at any crossed plane q[a] <= q[b]. For a dense
mass M, p <- p - 2*(n'M^-1 p)/(n'M^-1 n)*n. This preserves kinetic energy and
reverses exactly. The reflected domain uses raw log time constants, with no
log-gap transformation or gap Jacobian.
"""
import numpy as np
from numba import njit
from compiled_target import symmetric_gradient
PAIRS=np.array([[18,19],[19,20],[21,22]],dtype=np.int64)

@njit(cache=True,nogil=True)
def reflected_drift(q,p,inverse_mass,duration,pairs):
    x=q.copy();momentum=p.copy();remaining=duration;collisions=0
    for _ in range(1000):
        velocity=inverse_mass@momentum;hit=remaining+1.;which=-1
        for j in range(len(pairs)):
            a,b=pairs[j];closing=velocity[a]-velocity[b]
            if closing>0:
                time=max(0.,(x[b]-x[a])/closing)
                if time<=remaining and time<hit:hit=time;which=j
        if which<0:
            x+=remaining*velocity
            return x,momentum,collisions,True
        x+=hit*velocity;remaining-=hit;a,b=pairs[which];middle=.5*(x[a]+x[b]);x[a]=middle;x[b]=middle
        denominator=inverse_mass[a,a]+inverse_mass[b,b]-2*inverse_mass[a,b]
        correction=2*(velocity[a]-velocity[b])/denominator;momentum[a]-=correction;momentum[b]+=correction;collisions+=1
        if remaining<=1e-15:return x,momentum,collisions,True
    return x,momentum,collisions,False

@njit(cache=True,nogil=True)
def reflected_trajectory(position,momentum,step_size,n_steps,inverse_mass,args):
    x=position.copy();p=momentum.copy();value,g=symmetric_gradient(x,*args);p+=.5*step_size*g
    for i in range(n_steps):
        x,p,collisions,valid=reflected_drift(x,p,inverse_mass,step_size,PAIRS)
        if not valid:return x,p,-np.inf,False
        value,g=symmetric_gradient(x,*args)
        if not np.isfinite(value) or not np.isfinite(g).all():return x,p,-np.inf,False
        p+=(.5 if i==n_steps-1 else 1.)*step_size*g
    return x,-p,value,True

from effective_coordinates import effective_gradient
@njit(cache=True,nogil=True)
def effective_trajectory(position,momentum,step_size,n_steps,inverse_mass,args):
    x=position.copy();p=momentum.copy();value,g=effective_gradient(x,*args);p+=.5*step_size*g
    for i in range(n_steps):
        x,p,collisions,valid=reflected_drift(x,p,inverse_mass,step_size,PAIRS)
        if not valid:return x,p,-np.inf,False
        value,g=effective_gradient(x,*args)
        if not np.isfinite(value) or not np.isfinite(g).all():return x,p,-np.inf,False
        p+=(.5 if i==n_steps-1 else 1.)*step_size*g
    return x,-p,value,True
