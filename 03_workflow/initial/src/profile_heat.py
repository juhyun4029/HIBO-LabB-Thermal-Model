"""Conditional heat-allocation likelihood in unit-volume profile coordinates."""
import numpy as np
from ghk_heat import GHKHeatModel
from conditional_heat import linear_basis,all_fractions,valid_fractions
from profile_coordinates import ProfileModel,to_raw,prior
from global_transport import from_gap,to_gap,GAPS

class ProfileHeatModel(GHKHeatModel):
    to_raw=staticmethod(to_raw)
    prior_q=staticmethod(prior)
    def __init__(self,period):
        super().__init__(period);self.model=ProfileModel(period);self.base=self.model.base
    def basis(self,theta):
        if not np.isfinite(theta).all() or np.max(theta[GAPS])>20:raise ValueError('Invalid thermal proposal')
        v=np.zeros(55);v[:25]=theta;q=from_gap(v);p=self.base.native(to_raw(q));d=self.base.d
        if not np.isfinite(p).all() or np.max(abs(p[:25]))>100:raise ValueError('Thermal proposal outside numerical range')
        out=linear_basis(self.base.thermal_values(p),self.base.bi,self.base.wi,d['boundaries'],d['g_hvac'],d['tsupply'],d['gains'],d['ginf'],float(d['y'][0]),float(d['dt_seconds']),self.base.idx)
        if not np.isfinite(out).all():raise ValueError('Nonfinite basis')
        return out
    def q_from(self,theta,noise,a):
        v=np.zeros(55);v[:25]=theta;v[25:27]=noise;q=from_gap(v);u=to_raw(q);lf=-np.logaddexp(0,-u[1:14:2]);aa=all_fractions(a)
        q[27:]=(np.log(aa[:,:7]/aa[:,7,None])+lf[None,:]).ravel()
        return q
    def prior_a(self,theta,noise,a):
        if not valid_fractions(a):return -np.inf
        q=self.q_from(theta,noise,a)
        return float(prior(q,self.base.mean)-np.log(all_fractions(a)).sum()+theta[GAPS].sum())
    def from_q(self,q):
        theta=to_gap(q)[:25];noise=q[25:27].copy();a=self.base.alpha(to_raw(q))[:,:7].ravel();out=self.basis(theta);g=self.geometry(out,noise);z=self.z_from_a(a,g)
        lp,state=self.target_z(theta,noise,z,out,g)
        if state is None:raise ValueError('Unable to create profile state')
        return state
    def original_record(self,state,rng):
        q=self.q_from(state['theta'],state['noise'],state['a']);lp=self.model.logpost(q)
        return self.model.draw_labels(q,rng,lp),q
