"""17R8C with four independent heat-source allocation vectors.

Two backward-Euler substeps per one-minute E+ input interval. The algebraic
star solve avoids a dense matrix inverse and is equivalent to the full 8x8 solve.
Boundary mapping follows the actual IDF: ceiling/plenum; external/outdoors;
internal/neighbor; floor/ground. Q_HVAC=m*cp*(T_supply-T_model).
"""
import numpy as np
from numba import njit

@njit(cache=True,nogil=True)
def simulate(v,alpha,bi,wi,boundary,hvac_g,supply_t,gains,inf_g,initial_t,dt_seconds,substeps=2):
    dt=dt_seconds/substeps
    gb=1/v[0:14:2];gi=1/v[1:14:2];cd=v[17:24]/dt;air=v[24]/dt
    den=cd+gb+gi;a=cd/den;c=gi/den
    direct=np.zeros(4)
    for j in range(3):direct[wi[j]]+=1/v[14+j]
    dz=air+direct.sum()+np.sum(gi*(1-c))
    x=np.full(7,initial_t);z=initial_t;res=np.empty(len(hvac_g));forcing=np.empty(7);b=gi*a
    fb=gb/den;fa=np.empty((7,4));qa=np.empty(4)
    for j in range(4):
        qa[j]=alpha[j,7]
        for i in range(7):fa[i,j]=alpha[j,i]/den[i];qa[j]+=gi[i]*fa[i,j]
    fbd=direct.copy()
    for i in range(7):fbd[bi[i]]+=gi[i]*fb[i]
    for t in range(len(res)):
        for i in range(7):
            forcing[i]=fb[i]*boundary[t,bi[i]]
            for j in range(4):forcing[i]+=fa[i,j]*gains[t,j]
        F=hvac_g[t]*supply_t[t]+inf_g[t]*boundary[t,1]
        for j in range(4):F+=fbd[j]*boundary[t,j]+qa[j]*gains[t,j]
        inv=1/(dz+hvac_g[t]+inf_g[t])
        for _ in range(substeps):
            rhs=air*z+F
            for i in range(7):rhs+=b[i]*x[i]
            z=rhs*inv
            for i in range(7):x[i]=a[i]*x[i]+c[i]*z+forcing[i]
        res[t]=z
    return res

@njit(cache=True,nogil=True)
def simulate_fast(v,alpha,bi,wi,boundary,hvac_g,supply_t,gains,inf_g,initial_t,dt_seconds):
    dt=dt_seconds/2
    gb=1/v[0:14:2];gi=1/v[1:14:2];cd=v[17:24]/dt;air=v[24]/dt
    den=cd+gb+gi;a=cd/den;c=gi/den;b=gi*a;ba=b*a;fb=gb/den
    direct=np.zeros(4)
    for j in range(3):direct[wi[j]]+=1/v[14+j]
    f1=direct.copy();f2=direct.copy()
    for i in range(7):
        f1[bi[i]]+=gi[i]*fb[i];f2[bi[i]]+=(gi[i]+b[i])*fb[i]
    fq=alpha[:,:7].T/den[:,None]
    q1=alpha[:,7].copy();q2=q1.copy()
    for j in range(4):
        for i in range(7):q1[j]+=gi[i]*fq[i,j];q2[j]+=(gi[i]+b[i])*fq[i,j]
    dz=air+direct.sum()+np.sum(gi*(1-c));second_air=air+np.sum(b*c)
    result=np.empty(len(hvac_g));z=initial_t
    x0=initial_t;a0=a[0];aa0=a0*a0;b0=b[0];ba0=ba[0];c0=c[0];fb0=fb[0];bi0=bi[0]
    fq00=fq[0,0];fq01=fq[0,1];fq02=fq[0,2];fq03=fq[0,3]
    x1=initial_t;a1=a[1];aa1=a1*a1;b1=b[1];ba1=ba[1];c1=c[1];fb1=fb[1];bi1=bi[1]
    fq10=fq[1,0];fq11=fq[1,1];fq12=fq[1,2];fq13=fq[1,3]
    x2=initial_t;a2=a[2];aa2=a2*a2;b2=b[2];ba2=ba[2];c2=c[2];fb2=fb[2];bi2=bi[2]
    fq20=fq[2,0];fq21=fq[2,1];fq22=fq[2,2];fq23=fq[2,3]
    x3=initial_t;a3=a[3];aa3=a3*a3;b3=b[3];ba3=ba[3];c3=c[3];fb3=fb[3];bi3=bi[3]
    fq30=fq[3,0];fq31=fq[3,1];fq32=fq[3,2];fq33=fq[3,3]
    x4=initial_t;a4=a[4];aa4=a4*a4;b4=b[4];ba4=ba[4];c4=c[4];fb4=fb[4];bi4=bi[4]
    fq40=fq[4,0];fq41=fq[4,1];fq42=fq[4,2];fq43=fq[4,3]
    x5=initial_t;a5=a[5];aa5=a5*a5;b5=b[5];ba5=ba[5];c5=c[5];fb5=fb[5];bi5=bi[5]
    fq50=fq[5,0];fq51=fq[5,1];fq52=fq[5,2];fq53=fq[5,3]
    x6=initial_t;a6=a[6];aa6=a6*a6;b6=b[6];ba6=ba[6];c6=c[6];fb6=fb[6];bi6=bi[6]
    fq60=fq[6,0];fq61=fq[6,1];fq62=fq[6,2];fq63=fq[6,3]
    for t in range(len(result)):
        g0=gains[t,0];g1=gains[t,1];g2=gains[t,2];g3=gains[t,3]
        forcing=hvac_g[t]*supply_t[t]+inf_g[t]*boundary[t,1]
        inv=1/(dz+hvac_g[t]+inf_g[t])
        F1=forcing;F2=forcing
        for k in range(4):
            F1+=f1[k]*boundary[t,k]+q1[k]*gains[t,k]
            F2+=f2[k]*boundary[t,k]+q2[k]*gains[t,k]
        z1=(air*z+b0*x0+b1*x1+b2*x2+b3*x3+b4*x4+b5*x5+b6*x6+F1)*inv
        z=(second_air*z1+ba0*x0+ba1*x1+ba2*x2+ba3*x3+ba4*x4+ba5*x5+ba6*x6+F2)*inv
        x0=aa0*x0+(a0+1)*(fb0*boundary[t,bi0]+fq00*g0+fq01*g1+fq02*g2+fq03*g3)+c0*(a0*z1+z)
        x1=aa1*x1+(a1+1)*(fb1*boundary[t,bi1]+fq10*g0+fq11*g1+fq12*g2+fq13*g3)+c1*(a1*z1+z)
        x2=aa2*x2+(a2+1)*(fb2*boundary[t,bi2]+fq20*g0+fq21*g1+fq22*g2+fq23*g3)+c2*(a2*z1+z)
        x3=aa3*x3+(a3+1)*(fb3*boundary[t,bi3]+fq30*g0+fq31*g1+fq32*g2+fq33*g3)+c3*(a3*z1+z)
        x4=aa4*x4+(a4+1)*(fb4*boundary[t,bi4]+fq40*g0+fq41*g1+fq42*g2+fq43*g3)+c4*(a4*z1+z)
        x5=aa5*x5+(a5+1)*(fb5*boundary[t,bi5]+fq50*g0+fq51*g1+fq52*g2+fq53*g3)+c5*(a5*z1+z)
        x6=aa6*x6+(a6+1)*(fb6*boundary[t,bi6]+fq60*g0+fq61*g1+fq62*g2+fq63*g3)+c6*(a6*z1+z)
        result[t]=z
    return result
