"""Physical-coordinate prior density and derivatives."""
import numpy as np
from scipy.special import expit

def physical_prior_gradient(m,u):
    p=m.native(u);g=-(p[:25]-m.mean[:25])/m.sd[:25]**2;out=np.zeros(55);out[:25]=g
    for i in range(7):
        f=expit(u[2*i+1]);a=g[2*i];b=g[2*i+1];c=g[17+i]
        out[2*i]=a+b-c;out[2*i+1]=(1-f)*a-f*b+(2*f-1)*c;out[17+i]=c
    f=expit(u[15]);out[14]=-g[14]-g[15];out[15]=-(1-f)*g[14]+f*g[15]
    return out

