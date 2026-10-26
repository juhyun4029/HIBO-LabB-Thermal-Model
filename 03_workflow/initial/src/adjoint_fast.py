"""Compiled exact discrete adjoint with fast-math transformations disabled."""
import numpy as np
from numba import njit
@njit(cache=True,nogil=True)
def _effective_likelihood_gradient(v,alpha,bi,wi,boundary,hvac,supply,Q,inf,initial,y,idx,continuous,log_sigma,rho):
    n=len(y);air=v[24]/30.;X=np.empty((2*n+1,8));X[0,:]=initial;pred=np.empty(n)
    gb0=1/v[0];gi0=1/v[1];cd0=v[17]/30.;di0=1/(cd0+gb0+gi0);b0=gi0*di0;xi0=initial;bi0=bi[0]
    qa00=alpha[0,0]
    qa10=alpha[1,0]
    qa20=alpha[2,0]
    qa30=alpha[3,0]
    gb1=1/v[2];gi1=1/v[3];cd1=v[18]/30.;di1=1/(cd1+gb1+gi1);b1=gi1*di1;xi1=initial;bi1=bi[1]
    qa01=alpha[0,1]
    qa11=alpha[1,1]
    qa21=alpha[2,1]
    qa31=alpha[3,1]
    gb2=1/v[4];gi2=1/v[5];cd2=v[19]/30.;di2=1/(cd2+gb2+gi2);b2=gi2*di2;xi2=initial;bi2=bi[2]
    qa02=alpha[0,2]
    qa12=alpha[1,2]
    qa22=alpha[2,2]
    qa32=alpha[3,2]
    gb3=1/v[6];gi3=1/v[7];cd3=v[20]/30.;di3=1/(cd3+gb3+gi3);b3=gi3*di3;xi3=initial;bi3=bi[3]
    qa03=alpha[0,3]
    qa13=alpha[1,3]
    qa23=alpha[2,3]
    qa33=alpha[3,3]
    gb4=1/v[8];gi4=1/v[9];cd4=v[21]/30.;di4=1/(cd4+gb4+gi4);b4=gi4*di4;xi4=initial;bi4=bi[4]
    qa04=alpha[0,4]
    qa14=alpha[1,4]
    qa24=alpha[2,4]
    qa34=alpha[3,4]
    gb5=1/v[10];gi5=1/v[11];cd5=v[22]/30.;di5=1/(cd5+gb5+gi5);b5=gi5*di5;xi5=initial;bi5=bi[5]
    qa05=alpha[0,5]
    qa15=alpha[1,5]
    qa25=alpha[2,5]
    qa35=alpha[3,5]
    gb6=1/v[12];gi6=1/v[13];cd6=v[23]/30.;di6=1/(cd6+gb6+gi6);b6=gi6*di6;xi6=initial;bi6=bi[6]
    qa06=alpha[0,6]
    qa16=alpha[1,6]
    qa26=alpha[2,6]
    qa36=alpha[3,6]
    z=initial;direct=np.zeros(4)
    for k in range(3):direct[wi[k]]+=1/v[14+k]
    base=air+direct.sum()+(gi0-gi0*b0)+(gi1-gi1*b1)+(gi2-gi2*b2)+(gi3-gi3*b3)+(gi4-gi4*b4)+(gi5-gi5*b5)+(gi6-gi6*b6)
    for t in range(n):
        q0=Q[t,0];q1=Q[t,1];q2=Q[t,2];q3=Q[t,3]
        F=hvac[t]*supply[t]+inf[t]*boundary[t,1]
        for k in range(4):F+=direct[k]*boundary[t,k]+alpha[k,7]*Q[t,k]
        fi0=gb0*boundary[t,bi0]+qa00*q0+qa10*q1+qa20*q2+qa30*q3
        fi1=gb1*boundary[t,bi1]+qa01*q0+qa11*q1+qa21*q2+qa31*q3
        fi2=gb2*boundary[t,bi2]+qa02*q0+qa12*q1+qa22*q2+qa32*q3
        fi3=gb3*boundary[t,bi3]+qa03*q0+qa13*q1+qa23*q2+qa33*q3
        fi4=gb4*boundary[t,bi4]+qa04*q0+qa14*q1+qa24*q2+qa34*q3
        fi5=gb5*boundary[t,bi5]+qa05*q0+qa15*q1+qa25*q2+qa35*q3
        fi6=gb6*boundary[t,bi6]+qa06*q0+qa16*q1+qa26*q2+qa36*q3
        inv=1/(base+hvac[t]+inf[t])
        for sub in range(2):
            new=2*t+sub+1
            z=(air*z+F+b0*(cd0*xi0+fi0)+b1*(cd1*xi1+fi1)+b2*(cd2*xi2+fi2)+b3*(cd3*xi3+fi3)+b4*(cd4*xi4+fi4)+b5*(cd5*xi5+fi5)+b6*(cd6*xi6+fi6))*inv
            X[new,7]=z
            xi0=(cd0*xi0+fi0+gi0*z)*di0;X[new,0]=xi0
            xi1=(cd1*xi1+fi1+gi1*z)*di1;X[new,1]=xi1
            xi2=(cd2*xi2+fi2+gi2*z)*di2;X[new,2]=xi2
            xi3=(cd3*xi3+fi3+gi3*z)*di3;X[new,3]=xi3
            xi4=(cd4*xi4+fi4+gi4*z)*di4;X[new,4]=xi4
            xi5=(cd5*xi5+fi5+gi5*z)*di5;X[new,5]=xi5
            xi6=(cd6*xi6+fi6+gi6*z)*di6;X[new,6]=xi6
        pred[t]=z
    invs=np.exp(-2*log_sigma);gY=np.zeros(n);sse=0.;grho=0.;blocks=0
    for j in range(len(idx)):
        t=idx[j];e=pred[t]-y[t]
        if not continuous[j]:
            sse+=(1-rho*rho)*e*e;gY[t]-=(1-rho*rho)*e*invs;grho+=rho*e*e*invs;blocks+=1
        else:
            prev=idx[j-1];ep=pred[prev]-y[prev];inn=e-rho*ep
            sse+=inn*inn;gY[t]-=inn*invs;gY[prev]+=rho*inn*invs;grho+=inn*ep*invs
    ll=-.5*sse*invs-len(idx)*log_sigma+.5*blocks*np.log1p(-rho*rho)
    gls=sse*invs-len(idx);grho-=blocks*rho/(1-rho*rho)
    M=np.zeros((n,8))
    a0=0.
    a1=0.
    a2=0.
    a3=0.
    a4=0.
    a5=0.
    a6=0.
    a7=0.
    g0=0.
    g1=0.
    g2=0.
    g3=0.
    g4=0.
    g5=0.
    g6=0.
    g7=0.
    g8=0.
    g9=0.
    g10=0.
    g11=0.
    g12=0.
    g13=0.
    g14=0.
    g15=0.
    g16=0.
    g17=0.
    g18=0.
    g19=0.
    g20=0.
    g21=0.
    g22=0.
    g23=0.
    g24=0.
    for t in range(n-1,-1,-1):
        a7+=gY[t];inv=1/(base+hvac[t]+inf[t])
        for sub in range(1,-1,-1):
            old=2*t+sub;new=old+1
            mu7=(a7+b0*a0+b1*a1+b2*a2+b3*a3+b4*a4+b5*a5+b6*a6)*inv
            mu0=(a0+gi0*mu7)*di0
            mu1=(a1+gi1*mu7)*di1
            mu2=(a2+gi2*mu7)*di2
            mu3=(a3+gi3*mu7)*di3
            mu4=(a4+gi4*mu7)*di4
            mu5=(a5+gi5*mu7)*di5
            mu6=(a6+gi6*mu7)*di6
            g0+=mu0*gb0*(X[new,0]-boundary[t,bi0])
            g1+=gi0*(mu0-mu7)*(X[new,0]-X[new,7])
            g17+=mu0*cd0*(X[old,0]-X[new,0]);a0=cd0*mu0
            g2+=mu1*gb1*(X[new,1]-boundary[t,bi1])
            g3+=gi1*(mu1-mu7)*(X[new,1]-X[new,7])
            g18+=mu1*cd1*(X[old,1]-X[new,1]);a1=cd1*mu1
            g4+=mu2*gb2*(X[new,2]-boundary[t,bi2])
            g5+=gi2*(mu2-mu7)*(X[new,2]-X[new,7])
            g19+=mu2*cd2*(X[old,2]-X[new,2]);a2=cd2*mu2
            g6+=mu3*gb3*(X[new,3]-boundary[t,bi3])
            g7+=gi3*(mu3-mu7)*(X[new,3]-X[new,7])
            g20+=mu3*cd3*(X[old,3]-X[new,3]);a3=cd3*mu3
            g8+=mu4*gb4*(X[new,4]-boundary[t,bi4])
            g9+=gi4*(mu4-mu7)*(X[new,4]-X[new,7])
            g21+=mu4*cd4*(X[old,4]-X[new,4]);a4=cd4*mu4
            g10+=mu5*gb5*(X[new,5]-boundary[t,bi5])
            g11+=gi5*(mu5-mu7)*(X[new,5]-X[new,7])
            g22+=mu5*cd5*(X[old,5]-X[new,5]);a5=cd5*mu5
            g12+=mu6*gb6*(X[new,6]-boundary[t,bi6])
            g13+=gi6*(mu6-mu7)*(X[new,6]-X[new,7])
            g23+=mu6*cd6*(X[old,6]-X[new,6]);a6=cd6*mu6
            g14+=mu7*(X[new,7]-boundary[t,wi[0]])/v[14]
            g15+=mu7*(X[new,7]-boundary[t,wi[1]])/v[15]
            g16+=mu7*(X[new,7]-boundary[t,wi[2]])/v[16]
            g24+=mu7*air*(X[old,7]-X[new,7]);a7=air*mu7
            M[t,0]+=mu0
            M[t,1]+=mu1
            M[t,2]+=mu2
            M[t,3]+=mu3
            M[t,4]+=mu4
            M[t,5]+=mu5
            M[t,6]+=mu6
            M[t,7]+=mu7
    grad=np.array([g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g21,g22,g23,g24])
    ga=Q.T @ M
    return ll,grad,ga,gls,grho


@njit(cache=True,nogil=True)
def likelihood_gradient(v,alpha,bi,wi,boundary,hvac,supply,Q,inf,initial,y,idx,continuous,log_sigma_hour,rho_hour):
    # Hour-anchored stationary Gaussian AR(1) interpolated to five minutes.
    # Priors stay on hour parameters; there is no extra parameter Jacobian.
    fraction=5./60.
    phi=rho_hour**fraction
    log_sigma_five=log_sigma_hour+.5*(np.log1p(-phi*phi)-np.log1p(-rho_hour*rho_hour))
    ll,g,ga,gls,gphi=_effective_likelihood_gradient(v,alpha,bi,wi,boundary,hvac,supply,Q,inf,initial,y,idx,continuous,log_sigma_five,phi)
    dpdr=fraction*phi/rho_hour
    dlsdr=-phi*dpdr/(1.-phi*phi)+rho_hour/(1.-rho_hour*rho_hour)
    return ll,g,ga,gls,gphi*dpdr+gls*dlsdr
