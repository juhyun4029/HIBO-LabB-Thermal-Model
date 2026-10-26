"""Linear heat-allocation block and data-informed sampling coordinates.

For fixed thermal parameters, the minute-input SCM is affine in its 28
independent heat fractions. Four Dirichlet(1,...,1) priors define uniform
densities on the allocation simplexes. Gaussian regularization defines an
invertible coordinate map; thermal proposals receive a full-target
Metropolis-Hastings correction."""
import numpy as np
from numba import njit
from scipy.special import expit,softmax
from scipy.stats import truncnorm
from effective_coordinates import EffectiveModel, to_raw
from global_transport import from_gap, to_gap, GAPS
from tempered_hmc import effective_prior
from slice_transition import coordinate_slice


@njit(cache=True,nogil=True)
def linear_basis(v,bi,wi,boundary,hvac,supply,gains,inf,initial,dt_seconds,idx):
    dt=dt_seconds/2
    gb=1/v[0:14:2];gi=1/v[1:14:2];cd=v[17:24]/dt;air=v[24]/dt
    den=cd+gb+gi;a=cd/den;c=gi/den;b=gi*a;ba=b*a;fb=gb/den
    direct=np.zeros(4)
    for k in range(3):direct[wi[k]]+=1/v[14+k]
    f1=direct.copy();f2=direct.copy()
    for i in range(7):
        f1[bi[i]]+=gi[i]*fb[i];f2[bi[i]]+=(gi[i]+b[i])*fb[i]
    dz=air+direct.sum()+np.sum(gi*(1-c));second_air=air+np.sum(b*c)
    state=np.zeros((29,7));zone=np.zeros(29);state[0,:]=initial;zone[0]=initial
    out=np.empty((len(idx),29));pointer=0
    for t in range(len(hvac)):
        inv=1/(dz+hvac[t]+inf[t])
        base=hvac[t]*supply[t]+inf[t]*boundary[t,1]+gains[t,:].sum()
        F1=base;F2=base
        for k in range(4):F1+=f1[k]*boundary[t,k];F2+=f2[k]*boundary[t,k]
        for j in range(29):
            rhs1=air*zone[j];rhs2=0.
            for i in range(7):rhs1+=b[i]*state[j,i];rhs2+=ba[i]*state[j,i]
            if j==0:
                rhs1+=F1;rhs2+=F2
            else:
                k=(j-1)//7;i=(j-1)%7;gain=gains[t,k]
                rhs1+=(-1+gi[i]/den[i])*gain
                rhs2+=(-1+(gi[i]+b[i])/den[i])*gain
            z1=rhs1*inv;z2=(second_air*z1+rhs2)*inv
            for i in range(7):
                forcing=0.
                if j==0:forcing=fb[i]*boundary[t,bi[i]]
                elif (j-1)%7==i:forcing=gains[t,(j-1)//7]/den[i]
                state[j,i]=a[i]*a[i]*state[j,i]+(a[i]+1)*forcing+c[i]*(a[i]*z1+z2)
            zone[j]=z2
        if pointer<len(idx) and t==idx[pointer]:out[pointer,:]=zone;pointer+=1
    return out


def all_fractions(a):
    a=np.asarray(a).reshape(4,7)
    return np.column_stack([a,1-a.sum(1)])


def valid_fractions(a):
    return np.isfinite(a).all() and np.all(all_fractions(a)>0)


class ConditionalModel:
    def __init__(self,period):
        self.model=EffectiveModel(period);self.base=self.model.base;self.period=period
        # Mean/covariance of seven non-air entries of Dirichlet(1,...,1).
        self.mu0=np.full(28,1/8)
        cov=(8*np.eye(7)-np.ones((7,7)))/576
        self.L0=np.kron(np.eye(4),np.linalg.cholesky(cov))
        self.iL0=np.linalg.inv(self.L0)
        self.logdetL0=np.log(np.diag(self.L0)).sum()

    def q_from(self,theta,noise,a):
        v=np.zeros(55);v[:25]=theta;v[25:27]=noise;q=from_gap(v)
        raw=to_raw(q);lf=-np.logaddexp(0,-raw[1:14:2]);aa=all_fractions(a)
        q[27:]=(np.log(aa[:,:7]/aa[:,7,None])+lf[None,:]).ravel()
        return q

    def basis(self,theta):
        if not np.isfinite(theta).all() or np.max(theta[GAPS])>20:raise ValueError('Invalid thermal proposal')
        v=np.zeros(55);v[:25]=theta;q=from_gap(v);u=to_raw(q);p=self.base.native(u)
        if not np.isfinite(p).all() or np.max(np.abs(p[:25]))>100:raise ValueError('Thermal proposal outside numerical range')
        d=self.base.d
        out=linear_basis(self.base.thermal_values(p),self.base.bi,self.base.wi,d['boundaries'],d['g_hvac'],d['tsupply'],d['gains'],d['ginf'],float(d['y'][0]),float(d['dt_seconds']),self.base.idx)
        if not np.isfinite(out).all():raise ValueError('Nonfinite thermal basis')
        return out

    def whiten(self,out,noise):
        ls_hour,eta=self.base.lo+self.base.width*expit(noise);rho_hour=.98*expit(eta)
        rho,sigma_five=self.base.noise_at_five_minutes(ls_hour,rho_hour)
        ls=np.log(sigma_five)
        B=out[:,1:];yy=self.base.d['y'][self.base.idx]-out[:,0]
        Bw=B.copy();yw=yy.copy();cont=self.base.contiguous
        Bw[cont]=B[cont]-rho*B[np.flatnonzero(cont)-1]
        yw[cont]=yy[cont]-rho*yy[np.flatnonzero(cont)-1]
        Bw[~cont]*=np.sqrt(1-rho*rho);yw[~cont]*=np.sqrt(1-rho*rho)
        Bw*=np.exp(-ls);yw*=np.exp(-ls)
        return Bw,yw,ls,rho

    def geometry(self,out,noise):
        Bw,yw,ls,rho=self.whiten(out,noise)
        D=Bw@self.L0
        # Symmetric inverse square root is continuous through SVD sign changes.
        _,s,Vt=np.linalg.svd(D,full_matrices=False)
        invroot=1/np.sqrt(1+s*s);V=Vt.T
        L=self.L0@((V*invroot)@Vt)
        iL=((V/invroot)@Vt)@self.iL0
        mu=self.mu0+self.L0@(V@((Vt@(D.T@(yw-Bw@self.mu0)))/(1+s*s)))
        logdet=self.logdetL0-np.log1p(s*s).sum()/2
        # An interior mode in log-ratio coordinates. In fraction coordinates
        # its objective is the exact quadratic likelihood plus sum(log alpha),
        # the coordinate Jacobian. It is strictly concave. Its Laplace matrix
        # defines another bijection, never an approximate target density.
        H=Bw.T@Bw;h=Bw.T@yw;a=self.mu0.copy()
        def objective(v):
            ee=Bw@v-yw
            return .5*(ee@ee)-np.log(all_fractions(v)).sum()
        for it in range(100):
            aa=all_fractions(a);air=aa[:,7]
            grad=H@a-h-1/a+np.repeat(1/air,7)
            barrier=np.diag(1/a**2)
            for k in range(4):barrier[7*k:7*k+7,7*k:7*k+7]+=1/air[k]**2
            Hess=H+barrier;delta=np.linalg.solve(Hess,grad);decrement=grad@delta
            if decrement<1e-9:break
            rate=1.;old=objective(a)
            for _ in range(80):
                proposal=a-rate*delta
                if valid_fractions(proposal) and objective(proposal)<=old-1e-4*rate*max(decrement,0):break
                rate*=.5
            else:raise ValueError('Interior heat-mode line search failed')
            a=proposal
        else:raise ValueError('Interior heat-mode optimizer did not converge')
        aa=all_fractions(a);air=aa[:,7]
        eta=np.log(aa[:,:7]/air[:,None]).ravel();J=np.zeros((28,28))
        for k in range(4):
            ak=aa[k,:7];J[7*k:7*k+7,7*k:7*k+7]=np.diag(ak)-np.outer(ak,ak)
        barrier=np.diag(1/a**2)
        for k in range(4):barrier[7*k:7*k+7,7*k:7*k+7]+=1/air[k]**2
        Heta=J.T@(H+barrier)@J
        chol=np.linalg.cholesky((Heta+Heta.T)/2)
        Leta=np.linalg.solve(chol.T,np.eye(28));iLeta=chol.T
        return dict(Bw=Bw,yw=yw,mu=mu,L=L,iL=iL,logdet=float(logdet),eta=eta,Leta=Leta,iLeta=iLeta,logdet_eta=float(-np.log(np.diag(chol)).sum()),Lray=J@Leta,mode=a,mode_steps=it+1,ls=float(ls),rho=float(rho))

    def z_from_a(self,a,g):
        aa=all_fractions(a);eta=np.log(aa[:,:7]/aa[:,7,None]).ravel()
        return g['iLeta']@(eta-g['eta'])

    def likelihood(self,a,geometry):
        e=geometry['Bw']@a-geometry['yw'];m=self.base
        return float(-.5*(e@e)-len(m.idx)*geometry['ls']+.5*m.nblocks*np.log1p(-geometry['rho']**2))

    def prior_a(self,theta,noise,a):
        if not valid_fractions(a):return -np.inf
        q=self.q_from(theta,noise,a)
        pv=effective_prior(q,self.base.mean)[0]
        return float(pv-np.log(all_fractions(a)).sum()+theta[GAPS].sum())

    def target_z(self,theta,noise,z,out=None,geometry=None):
        try:
            if out is None:out=self.basis(theta)
            if geometry is None:geometry=self.geometry(out,noise)
            eta=geometry['eta']+geometry['Leta']@z
            aa=softmax(np.column_stack([eta.reshape(4,7),np.zeros(4)]),axis=1);a=aa[:,:7].ravel()
            pv=self.prior_a(theta,noise,a)
            if not np.isfinite(pv):return -np.inf,None
            ll=self.likelihood(a,geometry);lp=pv+ll+np.log(aa).sum()+geometry['logdet_eta']
            return float(lp),dict(theta=theta.copy(),noise=noise.copy(),z=z.copy(),a=a,out=out,g=geometry,lp=float(lp),ll=ll)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):return -np.inf,None

    def from_q(self,q):
        theta=to_gap(q)[:25];noise=q[25:27].copy();a=self.base.alpha(to_raw(q))[:,:7].ravel()
        out=self.basis(theta);g=self.geometry(out,noise);z=self.z_from_a(a,g)
        lp,state=self.target_z(theta,noise,z,out,g)
        if state is None:raise ValueError('Unable to construct conditional state')
        return state

    def stats(self,state):
        e=state['out'][:,0]+state['out'][:,1:]@state['a']-self.base.d['y'][self.base.idx];cont=self.base.contiguous
        cur=e[1:][cont[1:]];prev=e[:-1][cont[1:]];first=e[~cont]
        return np.array([e@e,cur@prev,prev@prev-first@first])

    def update_heat(self,state,rng,nsteps=40):
        a=state['a'].copy();g=state['g'];B=g['Bw'];H=B.T@B;h=B.T@g['yw']
        for _ in range(nsteps):
            v=g['Lray']@rng.normal(size=28)
            aa=all_fractions(a).ravel();vv=all_fractions(v).ravel()
            # all_fractions() appends 1-sum; a direction instead needs -sum.
            vv.reshape(4,8)[:,7]=-v.reshape(4,7).sum(1)
            pos=vv>0;neg=vv<0
            lo=float(np.max(-aa[pos]/vv[pos])) if pos.any() else -np.inf
            hi=float(np.min(-aa[neg]/vv[neg])) if neg.any() else np.inf
            precision=float(v@(H@v));slope=float(v@(H@a-h))
            if precision<=0:raise FloatingPointError('Nonpositive conditional ray curvature')
            center=-slope/precision;sd=1/np.sqrt(precision)
            t=truncnorm.rvs((lo-center)/sd,(hi-center)/sd,loc=center,scale=sd,random_state=rng)
            anew=a+t*v
            if not valid_fractions(anew):raise FloatingPointError('Truncated-normal draw left simplex')
            a=anew
        z=self.z_from_a(a,g)
        lp,out=self.target_z(state['theta'],state['noise'],z,state['out'],g)
        if out is None:raise FloatingPointError('Heat Gibbs update invalid')
        return out

    def update_noise(self,state,rng):
        stats=self.stats(state);x=state['noise'];fn=lambda v:self.base.noise_logpost(v,stats);lp=fn(x)
        for j in rng.permutation(2):x,lp,_,_=coordinate_slice(x,lp,fn,rng,[j],width=.4,max_steps=20)
        g=self.geometry(state['out'],x);z=self.z_from_a(state['a'],g)
        lp,out=self.target_z(state['theta'],x,z,state['out'],g)
        if out is None:raise FloatingPointError('Noise Gibbs update invalid')
        return out

    def original_record(self,state,rng):
        q=self.q_from(state['theta'],state['noise'],state['a'])
        lp=state['ll']+effective_prior(q,self.base.mean)[0]
        return self.model.draw_labels(q,rng,lp),q
