"""Export coherent joint posterior draws, diagnostics and minute fit errors."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from model import ROOT,Model,metrics
from diagnostic_parallel import diagnostic_definition, runtime_environment
from production_health import production_health


def _dump(path,payload):
    path.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')


def checkpoint_metrics(model,u,logp):
    if u.shape[1]==0:return dict(status='no_retained_draws')
    chain,draw=np.unravel_index(int(np.argmax(logp)),logp.shape)
    native=model.native(u[chain,draw])
    prediction=model.simulate_native(native)
    baseline=model.simulate_native(np.r_[np.log(model.ref),np.log(.3),1.5,np.zeros(28)])
    d=model.d
    return dict(representative='Highest original labeled log posterior among retained joint draws; no marginal-median parameter mixing',
                chain=int(chain+1),retained_draw=int(draw+1),
                training=metrics(prediction,d['y'],d['train']),
                validation=metrics(prediction,d['y'],d['valid']),
                baseline_training=metrics(baseline,d['y'],d['train']),
                baseline_validation=metrics(baseline,d['y'],d['valid']),
                metric_sampling_interval_seconds=60,conditioning_excluded=True)


def export_run(folder=None):
    folder=Path(folder or ROOT/'results/wide_case5')
    cp=folder/'checkpoint.npz'
    if not cp.exists():return dict(status='no_retained_checkpoint')
    with np.load(cp,allow_pickle=False) as ar:
        state=json.loads(str(ar['checkpoint_state_json']))
        u=ar['u'].copy();w=ar['w'].copy();logp=ar['logp'].copy()
        statistics=ar['statistics'].copy() if 'statistics' in ar.files else None
    if u.shape[1]==0:return dict(status='warmup_only_no_retained_draws')
    # Recompute health checks from every retained transition statistic.
    numerical_health=production_health(statistics)
    numerical_health['source']='checkpoint.npz:statistics (independently recomputed at export)'
    if statistics is not None and statistics.shape[:2]!=u.shape[:2]:
        numerical_health['blocking_failures'].append('Retained statistics axes do not match retained posterior states')
        numerical_health['passed']=False
    _dump(folder/'production_health.json',numerical_health)
    passed=state.get('N') is not None
    out=folder/'export';out.mkdir(exist_ok=True)
    if passed and not numerical_health['passed']:
        _dump(out/'summary.json',dict(status='NOT_CONVERGED_numerical_health_failed',
              N=state.get('N'),retained_per_chain=int(u.shape[1]),chains=int(u.shape[0]),
              numerical_health=numerical_health))
        raise AssertionError('Saved first-passing N failed independently recomputed production numerical health; do not label this export converged. '+ '; '.join(numerical_health['blocking_failures']))
    model=Model('july')
    config=json.loads((folder/'configuration.json').read_text())
    diag={}
    if u.shape[1]>=500:
        from reporting import diagnostics,canonical_diagnostics
        from canonical_model import to_v
        from profile_coordinates import to_raw
        diag=dict(diagnostic_definition=diagnostic_definition(), physical=diagnostics(model,u),ordered=canonical_diagnostics(to_v(to_raw(w))),
                  recomputed_retained_per_chain=int(u.shape[1]))
    if passed and not (diag.get('physical',{}).get('passed') and diag.get('ordered',{}).get('passed')):
        raise AssertionError('Saved first-passing N failed independently recomputed diagnostics; do not label this export converged.')
    fit=checkpoint_metrics(model,u,logp)
    c,d=fit['chain']-1,fit['retained_draw']-1
    representative=model.values(u[c,d])
    values=model.values(u)
    quantiles=np.quantile(values.reshape(-1,59),[.025,.5,.975],axis=0)
    perdiag={r['name']:r for r in diag.get('physical',{}).get('per_parameter',[])}
    with (out/'parameter_summary.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['parameter','unit','reference','median','p025','p975','representative_joint_draw','rhat','ess_bulk','ess_tail','ess_tail_lower_quantile','ess_tail_upper_quantile','diagnostic_schema']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for j,name in enumerate(model.names):
            row=dict(parameter=name,unit=('K/W' if j<17 else 'J/K' if j<25 else 'degC; 1-hour innovation reference' if j==25 else '1-hour correlation' if j==26 else 'fraction'),
                     reference=float(model.ref[j]) if j<25 else '',median=float(quantiles[1,j]),p025=float(quantiles[0,j]),p975=float(quantiles[2,j]),representative_joint_draw=float(representative[j]))
            row.update({k:v for k,v in perdiag.get(name,{}).items() if k in fields})
            row.update(ess_tail_lower_quantile=0.05, ess_tail_upper_quantile=0.95, diagnostic_schema='hibo-diagnostics-5-95-v1')
            writer.writerow(row)
    native=model.native(u[c,d]);prediction=model.simulate_native(native)
    baseline=model.simulate_native(np.r_[np.log(model.ref),np.log(.3),1.5,np.zeros(28)])
    with (out/'temperatures_1min.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['month','day','minute_end','split','EnergyPlus_C','joint_draw_model_C','IDF_reference_model_C'])
        for i in range(len(prediction)):
            split='Conditioning' if model.d['conditioning'][i] else 'Training' if model.d['train'][i] else 'Validation'
            writer.writerow([int(model.d['month'][i]),int(model.d['day'][i]),i%1440+1,split,float(model.d['y'][i]),float(prediction[i]),float(baseline[i])])
    # u and w retain chain/draw axes. Every row is a coherent joint state.
    np.savez_compressed(out/'raw_joint_chains.npz',u=u,w=w,logp=logp,
                        reported_physical=values,names=np.array(model.names),
                        first_passing_N=np.array(state.get('N') or -1),
                        latent_dimension=np.array(55),reported_dimension=np.array(59),
                        diagnostic_definition_json=np.array(json.dumps(diagnostic_definition())),
                        export_environment_json=np.array(json.dumps(runtime_environment())))
    _dump(out/'summary.json',dict(status='converged_at_first_N' if passed else 'NOT_CONVERGED_partial',
          N=state.get('N'),retained_per_chain=int(u.shape[1]),chains=4,free_R=17,free_C=8,
          independent_alpha=28,reported_alpha=32,independent_parameters=55,
          likelihood_interval_minutes=5,noise_prior_reference_minutes=60,
          train_likelihood_points=len(model.idx),configuration=config,diagnostics=diag,
          numerical_health=numerical_health,
          diagnostic_definition=diagnostic_definition(), export_environment=runtime_environment(),**fit))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[3,1]})
        x=np.arange(len(prediction))/1440+1
        sl=slice(None,None,10)
        axes[0].plot(x[sl],model.d['y'][sl],color='#777777',lw=.7,label='EnergyPlus')
        axes[0].plot(x[sl],prediction[sl],color='#216AA6',lw=.7,label='17R8C joint draw')
        axes[0].set_ylabel('Lab B air temperature (degC)');axes[0].legend(frameon=False,ncol=2)
        axes[1].plot(x[sl],(prediction-model.d['y'])[sl],color='#216AA6',lw=.6)
        axes[1].set(xlabel='July day (continuous model states)',ylabel='Error (degC)')
        for ax in axes:
            ax.axvspan(1,2,color='#999999',alpha=.15);ax.axvline(23,color='#333333',lw=.8,ls=':');ax.grid(alpha=.15)
        fig.tight_layout();fig.savefig(out/'temperature_fit.png',dpi=180);plt.close(fig)
    except ImportError:
        pass
    from variety_assessment import export_variety_assessment
    export_variety_assessment(model,u,out,representative_prediction=prediction,
                              baseline_prediction=baseline,
                              source_csv=ROOT/'inputs/HIBO_MCMC_1min.csv',
                              max_joint_draws=256,converged=passed)
    manifest={}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name!='EXPORT_MANIFEST_SHA256.json':
            manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    _dump(out/'EXPORT_MANIFEST_SHA256.json',manifest)
    return dict(status='export_complete',converged=passed,folder=str(out),retained_per_chain=int(u.shape[1]))


if __name__=='__main__':
    print(json.dumps(export_run(),indent=2))
