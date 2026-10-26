"""17R8C posterior with free thermal, heat-allocation, and AR(1) parameters.

IDF-derived references set independent lognormal priors for all 17 R and 8 C
values. Heat allocations use four eight-state simplexes. Noise parameters
specify the one-hour innovation standard deviation and correlation."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit, logit, softmax, logsumexp
from solver import simulate_fast as simulate

ROOT=Path(__file__).resolve().parents[1]

class Model:
    def __init__(self,period='july',data=None,reference=None):
        if period!='july':raise ValueError('This package runs July only; no annual or automatic second run.')
        d=dict(np.load(ROOT/'inputs/data.npz',allow_pickle=False)) if data is None else {k:np.array(v,copy=True) for k,v in data.items()}
        n=len(d['y'])
        if float(d['dt_seconds'])!=60:raise ValueError('Physical input interval must be 60 seconds.')
        if not np.all(d['month']==7):raise ValueError('Only July data are accepted.')
        conditioning=np.arange(n)<1440
        d['train']=(d['day']<=22)&~conditioning
        d['valid']=(d['day']>=23)&~conditioning
        d['conditioning']=conditioning
        self.d=d;self.period=period;self.likelihood_minutes=5.
        self.reference=json.loads((ROOT/'inputs/reference.json').read_text()) if reference is None else reference
        self.ref=np.r_[[v['reference'] for v in self.reference['physical_parameters']],self.reference['Czone']]
        states=['ceiling','extwall_1','extwall_2','extwall_3','intwall_1','intwall_2','floor','zone_air']
        self.names=[v['name'] for v in self.reference['physical_parameters']]+['zone_air:C','sigma_innovation_hour','rho_hour']+[f'alpha_{g}:{s}' for g in ['solar','lighting','occupant','equipment'] for s in states]
        self.ndim=55;self.mean=np.r_[np.log(self.ref)-.5*np.log(5),np.log(.3),1.5]
        self.sd=np.r_[np.full(25,np.sqrt(np.log(5))),1,1.5]
        self.lo=np.array([-5.,-5.]);self.width=np.array([7.,11.])
        comp=self.reference['components'];self.bi=np.array([v['boundary'] for v in comp[:7]],dtype=np.int64)
        self.wi=np.array([v['boundary'] for v in comp[7:]],dtype=np.int64)
        areas=np.array([v['area'] for v in comp[:7]]);self.weights=areas/areas.sum()
        self.idx=np.flatnonzero(d['train']&(np.arange(n)%5==4))
        self.contiguous=np.r_[False,np.diff(self.idx)==5];self.nblocks=int((~self.contiguous).sum())

    def initial(self):
        """Construct an initial state from the physical prior reference scales."""
        return self.inverse(np.r_[np.log(self.ref),np.log(.3),1.5,np.zeros(28)])

    @staticmethod
    def noise_at_five_minutes(log_sigma_hour,rho_hour):
        phi=rho_hour**(5./60.)
        sigma=np.exp(log_sigma_hour)*np.sqrt((1.-phi*phi)/(1.-rho_hour*rho_hour))
        return phi,sigma

    def inverse(self,p):
        u=np.array(p,copy=True,dtype=float)
        for i in range(7):
            a=p[...,2*i];b=p[...,2*i+1];t=np.logaddexp(a,b)
            u[...,2*i]=t;u[...,2*i+1]=a-b;u[...,17+i]=p[...,17+i]+a+b-t
        u[...,14]=np.logaddexp(-p[...,14],-p[...,15]);u[...,15]=p[...,15]-p[...,14]
        u[...,25:27]=logit(np.clip((p[...,25:27]-self.lo)/self.width,1e-12,1-1e-12))
        return u

    def native(self,u):
        p=np.array(u,copy=True,dtype=float)
        for i in range(7):
            t=u[...,2*i];r=u[...,2*i+1];lf=-np.logaddexp(0,-r);lg=-np.logaddexp(0,r)
            p[...,2*i]=t+lf;p[...,2*i+1]=t+lg;p[...,17+i]=u[...,17+i]-t-lf-lg
        p[...,14]=-u[...,14]+np.logaddexp(0,-u[...,15]);p[...,15]=-u[...,14]+np.logaddexp(0,u[...,15])
        p[...,25:27]=self.lo+self.width*expit(u[...,25:27])
        return p

    def values(self,u):
        p=self.native(u);v=np.exp(p[...,:27]);v[...,26]=.98*expit(p[...,26]);return np.concatenate([v,self.alpha(u).reshape((*u.shape[:-1],32))],axis=-1)

    def alpha(self,u):
        a=np.array(u[...,27:]).reshape((*u.shape[:-1],4,7))
        return softmax(np.concatenate([a,np.zeros((*a.shape[:-1],1))],axis=-1),axis=-1)

    def simulate_native(self,p):
        v=self.thermal_values(p);d=self.d
        return simulate(v,self.alpha(p),self.bi,self.wi,d['boundaries'],d['g_hvac'],d['tsupply'],d['gains'],d['ginf'],getattr(self,'initial_temperature',float(d['y'][0])),float(d['dt_seconds']))

    def thermal_values(self,p):
        v=np.exp(p[:25])
        v[17:25]/=getattr(self,'capacity_divisor',1.)
        return v

    def stats(self,p):
        pred=self.simulate_native(p)
        if not np.isfinite(pred).all():return None
        e=pred[self.idx]-self.d['y'][self.idx]
        cur=e[1:][self.contiguous[1:]];prev=e[:-1][self.contiguous[1:]];first=e[~self.contiguous]
        return np.array([e@e,cur@prev,prev@prev-first@first])

    def noise_logpost(self,un,stats):
        p=self.lo+self.width*expit(un);ls,eta=p;rho=.98*expit(eta)
        phi,sigma5=self.noise_at_five_minutes(ls,rho)
        q=stats[0]-2*phi*stats[1]+phi*phi*stats[2]
        ll=-.5*q/(sigma5*sigma5)-len(self.idx)*np.log(sigma5)+.5*self.nblocks*np.log1p(-phi*phi)
        prior=-.5*((ls-np.log(.3))**2+((eta-1.5)/1.5)**2)
        jac=np.sum(np.log(self.width)-np.logaddexp(0,-un)-np.logaddexp(0,un))
        return float(ll+prior+jac)

    def evaluate(self,u):
        if not np.isfinite(u).all():return -np.inf,None
        p=self.native(u)
        if np.max(abs(p[:25]))>80:return -np.inf,None
        stats=self.stats(p)
        if stats is None:return -np.inf,None
        prior=-.5*np.sum(((p[:25]-self.mean[:25])/self.sd[:25])**2)
        # Dirichlet(1,...,1) on each 8-simplex: constant density on fractions.
        # Add the log-ratio-coordinate Jacobian sum(log(alpha)).
        z=u[27:].reshape(4,7);z=np.column_stack([z,np.zeros(4)])
        alpha_jac=float(np.sum(z-logsumexp(z,axis=1)[:,None]))
        return float(prior+alpha_jac+self.noise_logpost(u[25:27],stats)),stats

    def logpost(self,u):return self.evaluate(u)[0]

    def gradient(self,u):
        from adjoint_fast import likelihood_gradient
        p=self.native(u);a=self.alpha(u);d=self.d
        if not np.isfinite(p).all() or np.max(abs(p[:25]))>80:return -np.inf,np.zeros(55)
        rho=.98*expit(p[26]);ls=p[25]
        ll,g,ga,gls,gr=likelihood_gradient(self.thermal_values(p),a,self.bi,self.wi,d['boundaries'],d['g_hvac'],d['tsupply'],d['gains'],d['ginf'],getattr(self,'initial_temperature',float(d['y'][0])),d['y'],self.idx,self.contiguous,ls,rho)
        prior=-.5*np.sum(((p[:25]-self.mean[:25])/self.sd[:25])**2)
        g-=((p[:25]-self.mean[:25])/self.sd[:25]**2)
        z=u[27:].reshape(4,7);z=np.column_stack([z,np.zeros(4)])
        aj=float(np.sum(z-logsumexp(z,axis=1)[:,None]))
        gp=np.zeros(55);gp[:25]=g
        for i in range(7):
            f=expit(u[2*i+1]);g1=g[2*i];g2=g[2*i+1];gc=g[17+i]
            gp[2*i]=g1+g2-gc;gp[2*i+1]=(1-f)*g1-f*g2+(2*f-1)*gc;gp[17+i]=gc
        f=expit(u[15]);g1=g[14];g2=g[15];gp[14]=-g1-g2;gp[15]=-(1-f)*g1+f*g2
        pn=-.5*((ls-np.log(.3))**2+((p[26]-1.5)/1.5)**2)
        b=expit(u[25:27]);jac=np.sum(np.log(self.width)-np.logaddexp(0,-u[25:27])-np.logaddexp(0,u[25:27]))
        ng=np.array([gls-(ls-np.log(.3)),gr*.98*expit(p[26])*(1-expit(p[26]))-(p[26]-1.5)/2.25])
        gp[25:27]=ng*self.width*b*(1-b)+1-2*b
        gp[27:]=(a[:,:7]*(ga[:,:7]-np.sum(ga*a,axis=1)[:,None])+1-8*a[:,:7]).reshape(-1)
        return float(ll+prior+aj+pn+jac),gp

def metrics(pred,y,mask):
    e=pred[mask]-y[mask]
    return dict(n=int(mask.sum()),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),bias=float(e.mean()))
