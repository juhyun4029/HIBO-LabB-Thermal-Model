"""Run all-pairs reduction and rule discovery from a fixed parent model.

Atomic state and checkpoint files support interruption and resume under the
same configuration. Validation data are used for candidate selection."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
import zipfile
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
from .topology import (build_initial_model, candidate_pairs, merge_pair, counts,
                       model_hash, apply_parameters, validate_model)
from .numerics import load_data, simulate, ReducedTarget
from .analysis import prediction_metrics, discover_rules
from .ranking import parameter_changes, rank_trials


SELECTION = dict(method='postfit_pareto', objectives=['d_R','d_C'],
                 min_pass_candidates=1,
                 tie_break=['validation_rmse','child_n_C','child_n_R','trial_id'])

RESCUE_POLICY = dict(enabled=True, max_attempts_per_trial=2,
                     trigger='no_passing_candidates',
                     restart='fresh_independent_chains',
                     authoritative='latest_predeclared_attempt',
                     sampler_overrides=dict(warmup=24000,settling=12000,max_draws=48000))


def jsonable(value):
    if isinstance(value, dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(jsonable(value),indent=2,ensure_ascii=False,allow_nan=False))
        stream.flush();os.fsync(stream.fileno())
    for attempt,delay in enumerate((.05,.1,.2,.4,.8,0)):
        try:
            os.replace(temp,path);break
        except PermissionError:
            if attempt==5:raise
            time.sleep(delay)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def load_config(root, path=None, smoke=False):
    config=read_json(path or Path(root)/'config.json')
    if smoke:
        config=deepcopy(config);config['mode']='smoke'
        config['sampler'].update(warmup=20,settling=20,min_draws=20,max_draws=20,
                                 check_every=10,checkpoint_every=10,smoke=True)
    if config.get('schema')!='hibo-reduction-allpairs-rca-v3': raise ValueError('Use the v3 configuration. Completed v2 results use reprocess, not resume.')
    if config.get('rescue')!=RESCUE_POLICY: raise ValueError('The predeclared fresh rescue policy must match this package.')
    if config.get('likelihood_minutes')!=5: raise ValueError('This package preserves the verified 5-minute likelihood.')
    if not 0<config['validation_rmse_limit_C']<=1: raise ValueError('Validation RMSE gate must be positive and <=1°C.')
    if config.get('mode') not in ('production','smoke'): raise ValueError('Unknown run mode')
    if not 1<=int(config['max_stages'])<=9: raise ValueError('max_stages must be 1–9')
    if config.get('selection')!=SELECTION: raise ValueError('This package uses the documented postfit Pareto definition and tie policy.')
    if 'fuzzy' in config: raise ValueError('Prefit fuzzy ranking is not used in this package.')
    return config


def default_output(root, mode='full'):
    """Deterministic output route; Windows results stay outside OneDrive.

    Explicit --output remains authoritative. Different extracted project paths
    get separate run directories; default status/resume/export use the same one.
    """
    root=Path(root).resolve()
    if os.name=='nt':
        base=os.environ.get('LOCALAPPDATA')
        if not base:raise RuntimeError('LOCALAPPDATA is unavailable. Specify --output outside OneDrive.')
        tag=hashlib.sha256(os.path.normcase(str(root)).encode()).hexdigest()[:12]
        return Path(base)/'HIBO_Reduction_AllPairs_RCA_v3'/tag/mode
    return root/'results'/mode


def initial_model(root):
    root=Path(root)
    estimates=read_json(root/'inputs/case5_representative_estimates.json')
    return build_initial_model(read_json(root/'inputs/reference.json'),estimates['estimates_by_parameter'])


def verify(root, output=None):
    """Verify input hashes and reconstruct the initial representative joint draw."""
    root=Path(root);manifest_path=root/'PACKAGE_MANIFEST_SHA256.json'
    manifest=read_json(manifest_path) if manifest_path.exists() else None
    mismatches=[]
    if manifest:
        for name,digest in manifest['files'].items():
            f=root/name
            if not f.is_file() or sha(f)!=digest: mismatches.append(name)
    if mismatches: raise ValueError('Package files changed: '+', '.join(mismatches)+'. Use a new documented package/configuration.')
    model=initial_model(root);data=load_data(root/'inputs/data.npz')
    if counts(model)['label']!='17R8C': raise ValueError('Initial topology is not 17R8C')
    if len(candidate_pairs(model))!=45: raise ValueError('All-pairs M0 must contain exactly 45 unordered envelope-component pairs.')
    start=time.perf_counter();prediction=simulate(model,data);seconds=time.perf_counter()-start
    with (root/'inputs/case5_reference_predictions.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    reference=np.array([float(r['joint_draw_model_C']) for r in rows])
    if reference.shape!=prediction.shape: raise ValueError('Reference prediction length mismatch')
    err=float(np.max(np.abs(prediction-reference)))
    metrics=prediction_metrics(prediction,data)
    expected=read_json(root/'inputs/case5_representative_estimates.json')['performance']['validation']['rmse']
    if err>1e-8 or abs(metrics['validation']['rmse']-expected)>1e-10:
        raise ValueError(f'M0 parity failed: max difference {err:g}, validation RMSE {metrics["validation"]["rmse"]}')
    report=dict(status='passed',model=counts(model),n_initial_pairs=len(candidate_pairs(model)),
                max_M0_temperature_difference_C=err,representative_validation_rmse_C=metrics['validation']['rmse'],
                source_expected_validation_rmse_C=expected,rows=len(prediction),simulation_seconds=seconds,
                python=sys.version,platform=platform.platform(),package_manifest_checked=manifest is not None,
                scope='Input integrity and deterministic model parity, not proof of reduced-model convergence.')
    if output: atomic_json(output,report)
    return report


def identity(root, config):
    root=Path(root)
    paths=sorted((root/'inputs').glob('*'))+sorted((root/'src/reduction').glob('*.py'))+[root/'run_reduction.py']
    return dict(config=config,files={str(p.relative_to(root)):sha(p) for p in paths if p.is_file()})


def _alive(pid):
    if pid<=0:return False
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return False
        try:
            code=ctypes.c_uint32()
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True


@contextmanager
def run_lock(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);p=output/'RUNNING.lock'
    if p.exists():
        old=read_json(p)
        if old.get('host')==socket.gethostname() and not _alive(int(old.get('pid',0))):p.unlink()
        else:raise RuntimeError(f'Another process may be using {output}; PID {old.get("pid")}.')
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),host=socket.gethostname(),started=time.time()),f)
    try:yield
    finally:p.unlink(missing_ok=True)


def plan(root, config, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=initial_model(root);atomic_json(output/'M_00_17R8C.json',model)
    rows=[]
    for left,right in candidate_pairs(model):
        child,specs,features=merge_pair(model,left,right)
        row=dict(left=left,right=right,child=counts(child)['label'],n_free=len(specs),**features)
        rows.append(row)
    atomic_json(output/'initial_pairs.json',rows)
    write_csv(output/'initial_pairs.csv',[dict(left=r['left'],right=r['right'],child=r['child'],n_free=r['n_free'],
                    pair_d_R=r['pair_d_R'],pair_d_C=r['pair_d_C']) for r in rows])
    return dict(model=counts(model),n_pairs=len(rows),output=str(output),
                note='Prefit descriptors only. Postfit d_R/d_C and Pareto ranks require each trial calibration.')


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(jsonable(v),ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def discover_all_rules(rows, config, output):
    """Steps 12–13: binary and numeric analyses run independently.

    Numeric discovery does not depend on PASS/FAIL contrast and never feeds
    back into this run's candidate generation, MCMC gates or Pareto selection.
    """
    from .numeric_rules import discover_numeric_rules
    output=Path(output)
    binary=discover_rules(rows,config['rules'],output/'binary')
    numeric=discover_numeric_rules(rows,config['numeric_rules'],output/'numeric')
    result=dict(status='completed',binary=binary,numeric=numeric,
                used_in_reduction_selection=False,
                scope='Exploratory postprocessing of accumulated trials; no guaranteed rule or external validation.')
    atomic_json(output/'summary.json',result)
    print(f"Steps 12–13 | binary: {binary.get('status')}; numeric: {numeric.get('status')}",flush=True)
    return result


def trial_seed(seed, trial_id):
    return int(hashlib.sha256(f'{seed}:{trial_id}'.encode()).hexdigest()[:8],16)


def physical_parameters(coordinates, specs):
    """Decode mixed unconstrained coordinates for archives and summaries.

    The authoritative target uses the same log/logit transforms. Alpha outputs
    are the actual merged allocation, not the fraction of the local budget.
    """
    values=np.asarray(coordinates,dtype=float)
    if values.shape[-1]!=len(specs):raise ValueError('Coordinate/spec dimension mismatch')
    physical=np.empty_like(values)
    for i,s in enumerate(specs):
        if s.get('transform','log')=='logit':
            physical[...,i]=float(s['budget'])*np.exp(-np.logaddexp(0.,-values[...,i]))
        elif s.get('transform','log')=='log':physical[...,i]=np.exp(values[...,i])
        else:raise ValueError('Unknown parameter transform')
    return physical


def save_trial_posterior(folder, result, specs, data, prediction, target=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    chains=np.asarray(result['chains']);lp=np.asarray(result['logp'])
    decode=target.physical_values if target is not None else lambda x:physical_parameters(x,specs)
    physical_chains=np.asarray(decode(chains))
    chosen=np.asarray(result['representative_log_values'])
    chosen_physical=np.asarray(decode(chosen))
    transforms=np.asarray([s.get('transform','log') for s in specs])
    arrays=dict(coordinates=chains,physical_parameters=physical_chains,
                coordinate_transforms=transforms,log_target=lp,
                names=np.asarray([s['name'] for s in specs]),
                representative_coordinates=chosen,representative_physical_values=chosen_physical)
    alpha_specs=[(i,s) for i,s in enumerate(specs) if s['field']=='alpha_fraction']
    if alpha_specs:
        air_alpha=np.zeros(chains.shape[:2]+(4,));merged_alpha=np.zeros_like(air_alpha)
        for i,s in alpha_specs:
            source=int(s['index']);merged_alpha[:,:,source]=physical_chains[:,:,i]
            air_alpha[:,:,source]=float(s['budget'])*np.exp(-np.logaddexp(0.,chains[:,:,i]))
        arrays.update(derived_air_alpha=air_alpha,merged_alpha=merged_alpha,
                      heat_source_names=np.asarray(['solar','lighting','occupant','equipment']))
    np.savez_compressed(folder/'posterior.npz',**arrays)
    physical=physical_chains.reshape(-1,chains.shape[-1]);qs=np.quantile(physical,[.025,.5,.975],axis=0)
    write_csv(folder/'parameter_summary.csv',[dict(parameter=s['name'],
        unit='fraction' if s['field']=='alpha_fraction' else 'J/K' if s['field']=='C' else 'K/W',
        transform=s.get('transform','log'),representative=chosen_physical[i],
        p025=qs[0,i],median=qs[1,i],p975=qs[2,i],
        prior_distribution='scaled_Beta' if s['field']=='alpha_fraction' else 'lognormal',
        prior_physical_mean=s['prior_mean'],prior_physical_sd=s.get('prior_sd',2*s['prior_mean']))
        for i,s in enumerate(specs)])
    derived=[]
    for i,s in enumerate(specs):
        if s['field']!='alpha_fraction':continue
        air_values=float(s['budget'])*np.exp(-np.logaddexp(0.,chains[:,:,i].ravel()));q=np.quantile(air_values,[.025,.5,.975])
        derived.append(dict(parameter='zone_air:'+s['name'].split(':')[-1],unit='fraction',
            status='derived_from_local_heat_conservation',representative=float(s['budget'])-chosen_physical[i],
            p025=q[0],median=q[1],p975=q[2],local_budget=s['budget']))
    if derived:
        write_csv(folder/'derived_air_alpha_summary.csv',derived)
        allocation=[]
        for i,s in alpha_specs:
            source=s['name'].split(':')[-1].removeprefix('alpha_')
            allocation.append(dict(source=source,component='merged_block',status='sampled',local_budget=s['budget'],
                representative=chosen_physical[i],p025=qs[0,i],median=qs[1,i],p975=qs[2,i]))
        for d in derived:
            allocation.append(dict(source=d['parameter'].split(':')[-1].removeprefix('alpha_'),
                component='zone_air',status='dependent_remainder_not_independent_parameter',
                local_budget=d['local_budget'],representative=d['representative'],p025=d['p025'],median=d['median'],p975=d['p975']))
        write_csv(folder/'alpha_allocation_summary.csv',allocation)
    np.savez_compressed(folder/'prediction.npz',predicted_C=prediction,EnergyPlus_C=data['y'],
                        train=data['train'],valid=data['valid'],minute_end=data['minute_end'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns=min(4,max(1,len(specs)));nrows=math.ceil(len(specs)/columns)
    fig,axs=plt.subplots(nrows,columns,figsize=(3.7*columns,2.9*nrows),squeeze=False)
    for i,ax in enumerate(axs.flat):
        if i>=len(specs):ax.set_visible(False);continue
        is_alpha=specs[i]['field']=='alpha_fraction'
        shown=physical[:,i] if is_alpha else chains[:,:,i].ravel()/np.log(10)
        selected=chosen_physical[i] if is_alpha else chosen[i]/np.log(10)
        ax.hist(shown,bins=40,density=True,color='#26669C',alpha=.8)
        ax.axvline(selected,color='#BD651F')
        ax.set_title(f"{specs[i]['name'].split(':')[-1]}: {chosen_physical[i]:.5g}")
        ax.set_xlabel('Heat allocation fraction' if is_alpha else 'log10 physical value')
        ax.set_ylabel('Density per fraction' if is_alpha else 'Density per log10 unit')
    fig.suptitle('Merged-block conditional posterior | orange = selected joint draw')
    fig.tight_layout();fig.savefig(folder/'posterior.png',dpi=160);plt.close(fig)


def archive(output, rows):
    output=Path(output);atomic_json(output/'all_trials.json',rows)
    flat=[]
    for row in rows:
        metrics=row.get('metrics',{});diagnostics=row.get('diagnostics',{})
        changes=row.get('parameter_changes',{})
        flat.append(dict(trial_id=row['trial_id'],depth=row['depth'],left=row['left'],right=row['right'],
            status=row['status'],parent=row['parent_label'],child=row['child_label'],
            pair_d_R=row['features'].get('pair_d_R'),pair_d_C=row['features'].get('pair_d_C'),
            d_R=changes.get('d_R'),d_C=changes.get('d_C'),pareto_rank=row.get('pareto_rank'),winner=row.get('winner',False),
            validation_rmse_C=metrics.get('validation',{}).get('rmse'),
            train_rmse_C=metrics.get('train',{}).get('rmse'),diagnostics=diagnostics,
            n_free=row['n_free'],n_free_RC=row.get('n_free_RC'),n_free_alpha=row.get('n_free_alpha'),
            authoritative_attempt=row.get('authoritative_attempt'),artifact_directory=row.get('artifact_directory'),
            elapsed_seconds=row.get('elapsed_seconds'),reason=row.get('reason')))
    write_csv(output/'all_trials.csv',flat)


def _attempt_name(attempt):
    return 'attempt_01_primary' if attempt==1 else 'attempt_02_rescue'


def _attempt_sampler_config(config, attempt):
    sampler=deepcopy(config['sampler'])
    if attempt==2:sampler.update(config['rescue']['sampler_overrides'])
    return sampler


def _run_attempt(root, output, state, config, data, parent, depth, parent_sha,
                 left, right, tid, pair_index, pair_count, attempt, previous=None):
    """Run an independently seeded attempt and commit an immutable result.
    
    A checkpoint resumes the same attempt. A rescue starts new chains, and its
    result supersedes the primary result without pooling their draws."""
    from .sampling import run_chains
    folder=Path(output)/'trials'/tid
    attempt_folder=folder/_attempt_name(attempt)
    result_file=attempt_folder/'result.json'
    if result_file.exists():
        row=read_json(result_file)
        if row['parent_hash']!=parent_sha or row.get('authoritative_attempt')!=attempt:
            raise ValueError('Committed attempt does not match its frozen parent or attempt index')
        return row,False
    child,specs,features=merge_pair(parent,left,right)
    attempt_folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'trial_definition.json',dict(trial_id=tid,parent_hash=parent_sha,parent=parent,
                trial=child,free_parameters=specs,features=features,
                inference='Merged R/C and local heat allocations; air shares derived by source conservation.'))
    target=ReducedTarget(child,specs,data,likelihood_minutes=config['likelihood_minutes'])
    sampler_config=_attempt_sampler_config(config,attempt)
    seed=trial_seed(config['seed'],f'{tid}:attempt_{attempt}')
    atomic_json(attempt_folder/'attempt_identity.json',dict(attempt=attempt,seed=seed,
                parent_hash=parent_sha,sampler_config=sampler_config,
                policy='Primary' if attempt==1 else 'Predeclared fresh independent rescue; authoritative even if worse'))
    def progress(info):
        status=dict(mode=config['mode'],stage=depth,pair=pair_index+1,pairs=pair_count,trial_id=tid,
                    attempt=attempt,attempt_label=_attempt_name(attempt),left=left,right=right,
                    minimum_pass_candidates=config['selection']['min_pass_candidates'],
                    elapsed_total_seconds=time.time()-state['started'],sampler=info)
        atomic_json(Path(output)/'status.json',status)
        phase=info.get('phase',info.get('status','sampling'))
        completed=info.get(phase+'_completed',info.get('production_completed',[]))
        detail='chain counts '+','.join(map(str,completed))
        latest=info.get('latest_diagnostics') or {};diag=latest.get('diagnostics') or {}
        if diag:detail+=f" | Rhat={diag.get('max_rhat')} bulkESS={diag.get('min_bulk_ess')} tailESS={diag.get('min_tail_ess')}"
        print(f"  {tid} | attempt {attempt} | {phase} | {detail} | elapsed {(time.time()-state['started'])/3600:.2f} h",flush=True)
    start=time.perf_counter()
    result=run_chains(target,attempt_folder/'mcmc',sampler_config,seed,resume=True,
                      stop_path=Path(output)/'STOP_REQUESTED',progress_callback=progress)
    if result['status']=='stopped':return None,True
    representative=np.asarray(result['representative_log_values'])
    prediction=target.predict(representative);metrics=prediction_metrics(prediction,data)
    converged=result['status']=='converged'
    accuracy=metrics['validation']['rmse']<=config['validation_rmse_limit_C']
    status=('pass' if accuracy else 'fail') if converged and config['mode']=='production' else 'unresolved'
    fitted=target.model_at(representative)
    attempts=deepcopy((previous or {}).get('attempts',[]))
    attempts=[a for a in attempts if a['attempt']<attempt]
    attempt_metadata=dict(attempt=attempt,seed=seed,status=status,sampler_status=result['status'],
        artifact_directory=str(attempt_folder.relative_to(output)),
        validation_rmse_C=metrics['validation']['rmse'],diagnostics=result['diagnostics'])
    attempts.append(attempt_metadata)
    row=dict(trial_id=tid,depth=depth,left=left,right=right,parent_hash=parent_sha,
        parent_label=counts(parent)['label'],child_label=counts(child)['label'],n_free=len(specs),
        n_free_RC=sum(s['field']!='alpha_fraction' for s in specs),
        n_free_alpha=sum(s['field']=='alpha_fraction' for s in specs),
        features=features,parameter_changes=parameter_changes(parent,fitted,left,right),
        pareto_rank=None,winner=False,status=status,sampler_status=result['status'],
        diagnostics=result['diagnostics'],metrics=metrics,
        reason='smoke_not_eligible' if config['mode']=='smoke' else ('not_converged' if not converged else 'criteria_passed' if accuracy else 'validation_RMSE_above_limit'),
        elapsed_seconds=time.perf_counter()-start,representative_logp=result.get('representative_logp'),
        representative_density_coordinates='natural log R/C and local-allocation logits; not physical MAP',
        fitted_model=fitted,authoritative_attempt=attempt,attempts=attempts,
        artifact_directory=str(attempt_folder.relative_to(output)),
        attempt_policy='Latest predeclared attempt is authoritative; no pooling and no best-attempt selection')
    atomic_json(attempt_folder/'fitted_model.json',fitted)
    atomic_json(attempt_folder/'parameter_changes.json',row['parameter_changes'])
    write_csv(attempt_folder/'parameter_changes.csv',row['parameter_changes']['per_parent'])
    write_csv(attempt_folder/'portwise_R_changes.csv',row['parameter_changes']['portwise_R_changes'])
    save_trial_posterior(attempt_folder,result,specs,data,prediction,target=target)
    atomic_json(attempt_folder/'mcmc_history.json',result.get('history',[]))
    # Commit last: a result.json always refers to a complete immutable attempt.
    atomic_json(result_file,row)
    return row,True


def _commit_trial(output, state, rows, row):
    rows[:]=[r for r in rows if r['trial_id']!=row['trial_id']]
    rows.append(row)
    folder=Path(output)/'trials'/row['trial_id']
    atomic_json(folder/'result.json',row)
    atomic_json(folder/'fitted_model.json',row['fitted_model'])
    state.update(rows=rows,status='running',updated=time.time())
    atomic_json(Path(output)/'workflow_state.json',state);archive(output,rows)


def run(root, config, output, resume=False, limit_trials=None, one_stage=False):
    root=Path(root);output=Path(output).resolve();fingerprint=identity(root,config);run_hash=stable_hash(fingerprint)
    print(f'Results folder: {output}',flush=True)
    minimum=config['selection']['min_pass_candidates']
    with run_lock(output):
        state_file=output/'workflow_state.json';stop_path=output/'STOP_REQUESTED'
        if state_file.exists():
            if not resume:raise RuntimeError('Run already exists. Use resume, or choose a new --output directory.')
            state=read_json(state_file)
            if state['run_hash']!=run_hash:raise ValueError('Resume refused: code, inputs or configuration changed.')
            if state.get('schema')!='hibo-workflow-allpairs-rca-v3':raise ValueError('Only unchanged v3 MCMC checkpoints can resume. Use reprocess for completed v2 results.')
            if state['status']=='completed':print('Already completed. No calculation restarted.',flush=True);return state
            stop_path.unlink(missing_ok=True)
            state.update(status='running',stop_reason=None,updated=time.time());atomic_json(state_file,state)
        else:
            if resume:raise RuntimeError('No run state exists to resume. Use run first.')
            if stop_path.exists():raise RuntimeError('Stop request exists in unused output. Choose a new output directory.')
            verification=verify(root,output/'preflight.json')
            state=dict(schema='hibo-workflow-allpairs-rca-v3',run_hash=run_hash,mode=config['mode'],status='running',
                       started=time.time(),updated=time.time(),current_model=initial_model(root),rows=[],path=[],depth=0,
                       rescue_waves={},verification=verification)
            atomic_json(output/'run_identity.json',fingerprint)
            atomic_json(output/'models/M_00_17R8C.json',state['current_model']);atomic_json(state_file,state)
        data=load_data(root/'inputs/data.npz');rows=state['rows'];completed_now=0
        print(f"Mode: {config['mode']}; current {counts(state['current_model'])['label']}; all component pairs; merged R/C/alpha sampled.",flush=True)
        print('Step 10: zero PASS stops; one PASS is adopted; multiple PASS use Pareto selection.',flush=True)
        if config['mode']=='smoke':print('SMOKE ONLY: production gates unchanged; no adoption or rescue authorized.',flush=True)
        def should_pause():
            if stop_path.exists():state.update(status='paused',stop_reason='stop_requested');return True
            if limit_trials is not None and completed_now>=limit_trials:
                state.update(status='paused',stop_reason='trial_budget_pause');return True
            return False
        try:
            while state['depth']<config['max_stages']:
                parent=state['current_model'];depth=state['depth'];parent_sha=model_hash(parent)
                pairs=candidate_pairs(parent)
                if not pairs:state.update(status='completed',stop_reason='no_component_pair');break
                print(f"Step 5 | M_{depth} {counts(parent)['label']} | {len(pairs)} unordered pairs",flush=True)
                records=[]
                for index,(left,right) in enumerate(pairs):
                    tid=f"k{depth:02d}_"+hashlib.sha256(f'{parent_sha}:{left}:{right}'.encode()).hexdigest()[:12]
                    records.append((index,left,right,tid))
                for index,left,right,tid in records:
                    existing=next((r for r in rows if r['trial_id']==tid),None)
                    if existing and existing['status'] in ('pass','fail','unresolved'):continue
                    if should_pause():break
                    row,sampled=_run_attempt(root,output,state,config,data,parent,depth,parent_sha,
                                              left,right,tid,index,len(pairs),1)
                    if row is None:state.update(status='paused',stop_reason='stop_requested');break
                    completed_now+=int(sampled);_commit_trial(output,state,rows,row)
                    print(f"  {tid} attempt 1: {row['status'].upper()} | validation RMSE={row['metrics']['validation']['rmse']:.4f}°C",flush=True)
                if state['status']=='paused':break
                ids={record[3] for record in records};wave=[r for r in rows if r['trial_id'] in ids]
                if len(wave)!=len(pairs):raise RuntimeError('Cannot adopt before all pairs from this parent are complete.')
                passing=sum(r['status']=='pass' for r in wave)
                # Freeze the complete rescue list before beginning. On resume,
                # finish it even if an early rescue has already supplied a PASS.
                key=str(depth);rescue=state['rescue_waves'].get(key)
                if rescue is None and config['mode']=='production' and passing<minimum:
                    unresolved=[r['trial_id'] for r in wave if r['status']=='unresolved']
                    if unresolved:
                        rescue=dict(parent_hash=parent_sha,trial_ids=unresolved,
                                    reason='no_passing_candidates',status='scheduled')
                        state['rescue_waves'][key]=rescue;atomic_json(state_file,state)
                if rescue is not None:
                    if rescue['parent_hash']!=parent_sha:raise ValueError('Rescue plan has a different frozen parent')
                    print(f"Step 8 | {passing} PASS; fresh rescue scheduled for {len(rescue['trial_ids'])} unresolved candidates",flush=True)
                    for index,left,right,tid in records:
                        if tid not in rescue['trial_ids']:continue
                        previous=next(r for r in rows if r['trial_id']==tid)
                        if previous.get('authoritative_attempt',1)>=2:continue
                        if should_pause():break
                        row,sampled=_run_attempt(root,output,state,config,data,parent,depth,parent_sha,
                                                  left,right,tid,index,len(pairs),2,previous=previous)
                        if row is None:state.update(status='paused',stop_reason='stop_requested');break
                        completed_now+=int(sampled);_commit_trial(output,state,rows,row)
                        print(f"  {tid} attempt 2: {row['status'].upper()} | validation RMSE={row['metrics']['validation']['rmse']:.4f}°C",flush=True)
                    if state['status']=='paused':break
                    rescue['status']='completed';atomic_json(state_file,state)
                if stop_path.exists():state.update(status='paused',stop_reason='stop_requested');break
                wave=[r for r in rows if r['trial_id'] in ids]
                wave=rank_trials(wave,min_pass_candidates=minimum)
                ranked_by_id={r['trial_id']:r for r in wave};rows[:]=[ranked_by_id.get(r['trial_id'],r) for r in rows]
                for row in wave:atomic_json(output/'trials'/row['trial_id']/'result.json',row)
                state.update(rows=rows,updated=time.time());atomic_json(state_file,state);archive(output,rows)
                passing=[r for r in wave if r['status']=='pass'];unresolved=any(r['status']=='unresolved' for r in wave)
                print(f'Step 10 | PASS {len(passing)}/{len(wave)}; one PASS is sufficient for adoption',flush=True)
                if not passing:
                    reason='no_passing_trial_with_unresolved' if unresolved else 'no_passing_trial'
                    state.update(status='completed',stop_reason=reason,
                                 comparison_limit=dict(n_candidates=len(wave),n_pass=len(passing),minimum_pass_candidates=minimum,
                                     note='No passing candidate after the completed evaluation policy. Unresolved candidates are not physical failures.'))
                    break
                winners=[r for r in passing if r.get('winner')]
                if len(winners)!=1:raise RuntimeError('Selection must return exactly one winner whenever at least one PASS exists')
                winner=winners[0]
                state['path'].append(dict(parent_depth=depth,trial_id=winner['trial_id'],
                    model=winner['child_label'],validation_rmse_C=winner['metrics']['validation']['rmse'],
                    d_R=winner['parameter_changes']['d_R'],d_C=winner['parameter_changes']['d_C'],
                    pareto_rank=winner['pareto_rank'],passing_candidates=len(passing),evaluated_candidates=len(wave),
                    selected_from_multiple_pass=len(passing)>1,unresolved_alternatives=sum(r['status']=='unresolved' for r in wave)))
                state.update(current_model=winner['fitted_model'],depth=depth+1,updated=time.time())
                atomic_json(output/f"models/M_{depth+1:02d}_{winner['child_label']}.json",state['current_model']);atomic_json(state_file,state)
                print(f"Step 11 | Adopt {winner['trial_id']} from {len(passing)} PASS candidates -> M_{depth+1} {winner['child_label']}",flush=True)
                if one_stage:state.update(status='paused',stop_reason='one_stage_requested');break
            else:state.update(status='completed',stop_reason='no_component_pair' if not candidate_pairs(state['current_model']) else 'configured_max_stages_reached')
        except KeyboardInterrupt:
            state.update(status='paused',stop_reason='keyboard_interrupt')
            print('Interrupted. Completed checkpoint chunks can be resumed.',flush=True)
        except Exception:
            state.update(status='error',stop_reason='unexpected_error')
            (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise
        finally:
            state.update(updated=time.time(),rows=rows);atomic_json(state_file,state);archive(output,rows)
            atomic_json(output/'selected_model.json',state['current_model']);atomic_json(output/'selected_path.json',state['path'])
        rule_result=discover_all_rules(rows,config,output/'rules')
        atomic_json(output/'summary.json',dict(status=state['status'],stop_reason=state.get('stop_reason'),mode=config['mode'],
                    n_trials=len(rows),n_attempts=sum(len(r.get('attempts',[])) for r in rows),
                    n_pass=sum(r['status']=='pass' for r in rows),n_fail=sum(r['status']=='fail' for r in rows),
                    n_unresolved=sum(r['status']=='unresolved' for r in rows),
                    current_model=counts(state['current_model']),accepted_steps=state['depth'],
                    minimum_pass_candidates=minimum,comparison_limit=state.get('comparison_limit'),
                    elapsed_seconds=time.time()-state['started'],rule_discovery=rule_result,
                    scope='All-pairs greedy HIBO July conditional calibration; a sole PASS is eligible. Not a global minimum or externally validated rule set.'))
        print(f"{state['status'].upper()}: {state.get('stop_reason')}; {counts(state['current_model'])['label']}",flush=True)
        print(f'Results folder: {output}',flush=True)
        return state


def export_results(root, output, destination=None):
    root=Path(root);output=Path(output)
    if not (output/'workflow_state.json').is_file():raise ValueError('No results found')
    if (output/'RUNNING.lock').exists():raise RuntimeError('Stop the run before exporting a coherent result snapshot.')
    destination=Path(destination or root/f"HIBO_Reduction_AllPairs_RCA_v3_results_{output.name}_{time.strftime('%Y%m%d_%H%M%S')}.zip")
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix!='.tmp' and p.name!='STOP_REQUESTED':z.write(p,Path('results')/p.relative_to(output))
        z.write(root/'config.json','package_config.json')
        z.write(root/'README_KO.md','README_KO.md')
    return str(destination)
