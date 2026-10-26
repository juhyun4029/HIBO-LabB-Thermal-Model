"""Run initial calibration, archive its posterior, then reduce the model and extract rules.

Manage stage sequencing, input provenance, result archives, and checkpoint resume.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'pipeline_state.json'
LOCK = ROOT / 'PIPELINE.lock'
STOP = ROOT / 'PIPELINE_STOP_REQUESTED'
PACKAGE_ID = 'HIBO_LabB_SingleSP_FullPipeline_v3'

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()

def read(path, default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def sha(path):
    """Compute SHA-256, retrying transient reads with smaller unbuffered blocks.

    Each retry starts at byte zero. Persistent errors include the failing offset.
    """
    path=Path(path)
    failures=[];last_error=None
    for block_size in (1024*1024,64*1024,8*1024):
        h=hashlib.sha256();offset=0
        try:
            with path.open('rb',buffering=0) as f:
                while True:
                    chunk=f.read(block_size)
                    if not chunk:break
                    h.update(chunk);offset+=len(chunk)
            return h.hexdigest()
        except OSError as error:
            last_error=error
            failures.append({'chunk_bytes':block_size,'offset':offset,'errno':error.errno,
                             'winerror':getattr(error,'winerror',None),'message':str(error)})
    raise RuntimeError('Cannot read package/result file: '+str(path.resolve())+
        '\nRead attempts: '+json.dumps(failures)+
        '\nExtract the complete ZIP to a new local folder outside OneDrive, then run SETUP there. '
        'No integrity check was skipped; the current operation was stopped.') from last_error

class RunLock:
    def __enter__(self):
        self.f=LOCK.open('a+b');self.f.seek(0,2)
        if not self.f.tell():self.f.write(b' ');self.f.flush()
        self.f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.f.close();raise RuntimeError('This pipeline is already active. Use status or stop; no second run started.')
        return self
    def __exit__(self,*args):
        self.f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.f.fileno(),fcntl.LOCK_UN)
        self.f.close()

def active():
    try:
        with RunLock():return False
    except RuntimeError:return True

def verify_package():
    manifest=read(ROOT/'MANIFEST_SHA256.json')
    if not manifest:raise RuntimeError('Package manifest missing; extract the complete ZIP.')
    for name,digest in manifest.items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or sha(p)!=digest:
            raise RuntimeError('Missing/modified package input or code: '+name)
    return sha(ROOT/'MANIFEST_SHA256.json')

def save_state(s,stage=None,**extra):
    if stage:s['stage']=stage
    s.update(extra,updated_utc=now());atomic_json(STATE,s)
    if s.get('workdir'):atomic_json(Path(s['workdir'])/'pipeline_state.json',s)

def paths(s):
    w=Path(s['workdir'])
    return w/'initial',w/'reduction',w/'reduction_results'

def request_stop(s):
    STOP.write_text('Graceful stop requested\n',encoding='utf-8')
    initial,reduction,out=paths(s)
    if s.get('stage')=='initial_running':
        (initial/'CASE5_STOP_REQUESTED').write_text('Stop at a coherent checkpoint\n',encoding='utf-8')
    if s.get('stage')=='reduction_running':
        out.mkdir(exist_ok=True);(out/'STOP_REQUESTED').write_text('Stop at a coherent checkpoint\n',encoding='utf-8')

def child(s,script,*args):
    script=Path(script);env=os.environ.copy()
    env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    cmd=[sys.executable,'-u',str(script),*map(str,args)]
    print('\nRunning: '+' '.join(cmd),flush=True)
    log=Path(s['workdir'])/'pipeline_console.log'
    with log.open('a',encoding='utf-8',buffering=1) as f:
        f.write('\n'+now()+' '+repr(cmd)+'\n')
        process=subprocess.Popen(cmd,cwd=script.parent,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        s['child_pid']=process.pid;save_state(s)
        while True:
            try:
                line=process.stdout.readline()
                if line:print(line,end='',flush=True);f.write(line)
                elif process.poll() is not None:break
            except KeyboardInterrupt:
                request_stop(s)
                print('Stop requested. Waiting for saved checkpoints and result export; keep this terminal open.',flush=True)
        code=process.wait()
    s.pop('child_pid',None);save_state(s)
    return code

def zipped(destination, entries):
    """Commit only after ZIP CRC validation. Archive failure blocks handoff."""
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_name(destination.name+'.tmp')
    # Windows _commit/os.fsync requires a writable descriptor.  Keep that
    # descriptor open until ZipFile has written its central directory.
    with tmp.open('w+b') as f:
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
            for source,name in entries:
                source=Path(source)
                if not source.is_file():
                    raise FileNotFoundError('Required archive input is missing: '+str(source))
                z.write(source,str(name).replace('\\','/'))
        f.flush();os.fsync(f.fileno())
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Result ZIP verification failed: '+bad)
    os.replace(tmp,destination)
    return dict(path=str(destination),sha256=sha(destination),bytes=destination.stat().st_size,saved_utc=now(),crc_verified=True)

def result_files(folder,prefix):
    if not Path(folder).exists():return []
    return [(p,Path(prefix)/p.relative_to(folder)) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and not p.name.endswith(('.tmp','.tmp.npz','.lock'))
            and '__pycache__' not in p.parts and not any(x.startswith('.') for x in p.relative_to(folder).parts)
            and p.name!='STOP_REQUESTED']

def archive_initial(s):
    initial,_,_=paths(s)
    entries=result_files(initial/'results','initial/results')
    entries+=result_files(initial/'src','initial/src')
    for name in ('data.npz','reference.json','metadata.json','simulation_audit.json','HIBO.idf','build_data.py'):
        entries.append((initial/'inputs'/name,'initial/inputs/'+name))
    for name in ('CASE5_SETTINGS.json','run_case5.py','PACKAGE_MANIFEST_SHA256.json','requirements.txt','SAMPLER_DESIGN.md'):
        entries.append((initial/name,'initial/'+name))
    entries.extend([(ROOT/'pipeline_state.json','pipeline_state_at_archive.json'),(ROOT/'README_KO.md','README_KO.md')])
    receipt=zipped(ROOT/'exports/Initial_MCMC_results.zip',entries)
    save_state(s,'initial_archived',initial_archive=receipt)
    atomic_json(ROOT/'exports/Initial_MCMC_archive_receipt.json',receipt)
    print('\nINITIAL RESULT ZIP SAVED AND VERIFIED: '+receipt['path'],flush=True)
    return receipt

def initial_gate(initial):
    """Validate the completed posterior export before reduction.

    Settling diagnostics and retained-posterior diagnostics are reported
    separately; an incomplete initial fit cannot enter reduction.
    """
    folder=Path(initial)/'results/single_sp_initial'
    status=read(folder/'status.json',{})
    stage=status.get('stage','not_started')
    summary=read(folder/'export/summary.json',{})
    bundle=read(folder/'result_bundle.json',{})
    reasons=[]
    gate=dict(passed=False,reasons=reasons,initial_sampler_stage=stage,
              first_passing_N=summary.get('N'),checked_utc=now())
    if stage!='complete':
        reasons.append('initial sampler stage is '+str(stage))
        settling=read(folder/'settling_diagnostics.json',{})
        if settling:
            keys=('max_rhat','min_bulk_ess','min_tail_ess','invalid_parameters')
            gate['last_settling_diagnostics']=dict(
                source='settling_diagnostics.json',role='discarded fixed-kernel readiness window; not retained posterior',
                completed=settling.get('completed'),
                assessment_start_iteration=settling.get('assessment_start_iteration'),
                assessment_end_iteration=settling.get('assessment_end_iteration'),
                physical={k:settling.get('physical',{}).get(k) for k in keys},
                ordered={k:settling.get('ordered',{}).get(k) for k in keys},
                blocking_failures=settling.get('blocking_failures',[]),
                chain_health=settling.get('chain_health',[]))
            if stage=='settling_gate_failed':
                reasons.extend(settling.get('blocking_failures',[]))
        if stage=='settling_gate_failed':
            gate['production_diagnostics_status']='not_available_no_retained_posterior'
            reasons.append('No retained posterior was produced; production diagnostic/export checks were not evaluated.')
        else:
            gate['production_diagnostics_status']='not_authorized_until_complete'
            if status.get('explanation'):reasons.append(status['explanation'])
        return gate
    if not summary:
        reasons.append('Initial sampler completed, but its posterior export summary is missing; handoff is blocked.')
        gate['production_diagnostics_status']='export_summary_missing'
        return gate
    gate['production_diagnostics_status']='evaluated_from_completed_export'
    if summary.get('status')!='converged_at_first_N':reasons.append('export does not confirm first-passing convergence')
    if not isinstance(summary.get('N'),int) or summary.get('N',0)<=0:reasons.append('no positive first-passing N')
    if summary.get('independent_parameters')!=55 or summary.get('chains')!=4:reasons.append('unexpected model dimensions or chains')
    numerical=summary.get('numerical_health')
    if not isinstance(numerical,dict) or 'passed' not in numerical:
        reasons.append('Production numerical health is missing from the independently recomputed export')
    elif numerical.get('passed') is not True or numerical.get('blocking_failures'):
        reasons.append('Production numerical health did not pass; reduction handoff is blocked')
        reasons.extend(numerical.get('blocking_failures',[]))
    gate['numerical_health']=numerical
    for family in ('physical','ordered'):
        d=summary.get('diagnostics',{}).get(family,{})
        if not d:
            reasons.append(family+' production diagnostics are missing')
            continue
        limits=(('max_rhat',lambda v:v<1.10),('min_bulk_ess',lambda v:v>=400),('min_tail_ess',lambda v:v>=400))
        if not d.get('passed') or d.get('invalid_parameters',0):reasons.append(family+' production diagnostics not passed')
        for key,test in limits:
            v=d.get(key)
            if not isinstance(v,(int,float)) or not math.isfinite(v):
                reasons.append(family+' '+key+' is missing or nonfinite')
            elif not test(v):reasons.append(family+' '+key+' fails: '+str(v))
    if bundle.get('posterior_export',{}).get('status')!='export_complete':reasons.append('posterior export incomplete')
    for name in ('parameter_summary.csv','temperatures_1min.csv','raw_joint_chains.npz'):
        if not (folder/'export'/name).is_file():reasons.append('missing '+name)
    cp=folder/'checkpoint.npz'
    if cp.is_file():
        import numpy as np
        with np.load(cp,allow_pickle=False) as a:
            checkpoint=json.loads(str(a['checkpoint_state_json']))
        if checkpoint.get('N')!=summary.get('N') or checkpoint.get('N') is None:
            reasons.append('checkpoint/export first-passing N mismatch')
    else:reasons.append('missing retained checkpoint')
    manifest=read(folder/'export/EXPORT_MANIFEST_SHA256.json',{})
    for name in ('summary.json','parameter_summary.csv','temperatures_1min.csv'):
        f=folder/'export'/name
        if not f.is_file() or manifest.get(name)!=sha(f):reasons.append('missing/stale export checksum: '+name)
    gate['passed']=not reasons
    return gate

def prepare_handoff(initial,reduction,receipt):
    """Transfer all 59 reported values from one retained joint draw."""
    initial=Path(initial);reduction=Path(reduction);out=initial/'results/single_sp_initial/export'
    if sha(receipt['path'])!=receipt['sha256'] or not receipt.get('crc_verified'):
        raise RuntimeError('Initial result ZIP is missing or changed; reduction cannot start.')
    summary=read(out/'summary.json');target=reduction/'inputs';target.mkdir(exist_ok=True)
    with (out/'parameter_summary.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    vals={r['parameter']:dict(base_value=float(r['representative_joint_draw']),unit=r['unit']) for r in rows}
    if len(vals)!=59 or not all(math.isfinite(x['base_value']) for x in vals.values()):raise ValueError('Invalid representative parameter export')
    estimate=dict(schema='new-single-sp-M0-joint-draw-v1',estimates_by_parameter=vals,
        performance={k:summary[k] for k in ('training','validation')},
        selection=summary['representative'],chain=summary['chain'],retained_draw=summary['retained_draw'],
        initial_result_archive=receipt,old_case5_parameter_values_used=False)
    for name in ('data.npz','reference.json','metadata.json','simulation_audit.json','HIBO.idf'):
        shutil.copy2(initial/'inputs'/name,target/name)
    for src,dest in [('summary.json','case5_summary.json'),('parameter_summary.csv','case5_parameter_summary.csv'),('temperatures_1min.csv','case5_reference_predictions.csv')]:
        shutil.copy2(out/src,target/dest)
    atomic_json(target/'case5_representative_estimates.json',estimate)
    atomic_json(target/'case5_diagnostics.json',summary['diagnostics'])
    atomic_json(target/'handoff.json',dict(created_utc=now(),new_data_sha256=sha(target/'data.npz'),
        source_summary_sha256=sha(out/'summary.json'),source_parameter_summary_sha256=sha(out/'parameter_summary.csv'),
        representative_chain=summary['chain'],representative_retained_draw=summary['retained_draw'],
        initial_archive_sha256=receipt['sha256'],
        legacy_filename_note='case5_* filenames are compatibility names; contents are entirely from this new single-setpoint M0.'))
    files=list(target.glob('*'))+list((reduction/'src/reduction').glob('*.py'))+[reduction/'run_reduction.py',reduction/'config.json']
    atomic_json(reduction/'PACKAGE_MANIFEST_SHA256.json',{'files':{str(p.relative_to(reduction)).replace('\\','/'):sha(p) for p in files if p.is_file()}})

def archive_reduction(s):
    initial,reduction,out=paths(s)
    entries=result_files(out,'reduction/results')+result_files(reduction/'inputs','reduction/inputs')+result_files(reduction/'src','reduction/src')
    for name in ('run_reduction.py','config.json','PACKAGE_MANIFEST_SHA256.json','SCIENTIFIC_METHOD.md'):
        entries.append((reduction/name,'reduction/'+name))
    entries.extend([(ROOT/'pipeline_state.json','pipeline_state.json'),(ROOT/'README_KO.md','README_KO.md')])
    receipt=zipped(ROOT/'exports/Reduction_and_Rules_results.zip',entries)
    save_state(s,reduction_archive=receipt)
    # A final combined bundle references the independently preserved M0 ZIP.
    all_receipt=zipped(ROOT/'exports/FullPipeline_results.zip',[
        (ROOT/'exports/Initial_MCMC_results.zip','Initial_MCMC_results.zip'),
        (ROOT/'exports/Reduction_and_Rules_results.zip','Reduction_and_Rules_results.zip'),
        (ROOT/'pipeline_state.json','pipeline_state.json'),(ROOT/'README_KO.md','README_KO.md')])
    save_state(s,full_archive=all_receipt)
    print('FINAL/PAUSED RESULT ZIP: '+all_receipt['path'],flush=True)

def proceed(s,resume=False):
    initial,reduction,out=paths(s)
    # Validate working inputs and source against the package manifest.
    for name,digest in read(ROOT/'initial/PACKAGE_MANIFEST_SHA256.json',{}).items():
        p=initial/name
        if not p.is_file() or sha(p)!=digest:
            raise RuntimeError('Initial working copy changed: '+name)
    folder=initial/'results/single_sp_initial'
    stage=read(folder/'status.json',{}).get('stage')
    terminal=('complete','settling_gate_failed','search_budget_reached')
    if stage not in terminal:
        action='resume' if (folder/'configuration.json').is_file() else 'start'
        if folder.exists() and any(folder.iterdir()) and action=='start':
            raise RuntimeError('Initial setup was interrupted before a resumable configuration. Existing diagnostics are preserved; use a fresh extracted folder.')
        save_state(s,'initial_running')
        code=child(s,initial/'run_case5.py',action)
        save_state(s,initial_exit_code=code)
    # This archive always precedes the gate and every reduction operation.
    if not s.get('initial_archive') or s.get('stage') in ('initial_running','failed','paused_initial'):
        archive_initial(s)
    gate=initial_gate(initial);save_state(s,initial_gate=gate)
    if STOP.exists():
        save_state(s,'paused_initial');print('Paused after saving initial results.');return
    if not gate['passed']:
        current=read(folder/'status.json',{}).get('stage')
        save_state(s,'initial_not_converged' if current in terminal else 'paused_initial')
        print('Reduction NOT started. Initial results were saved. '+ '; '.join(gate['reasons']),flush=True)
        if gate.get('last_settling_diagnostics'):
            print('SAVED SETTLING DIAGNOSTICS (discarded, not posterior): '+json.dumps(gate['last_settling_diagnostics']),flush=True)
        return
    if not (out/'workflow_state.json').is_file():
        save_state(s,'handoff')
        prepare_handoff(initial,reduction,s['initial_archive'])
        if child(s,reduction/'run_reduction.py','verify')!=0:raise RuntimeError('Initial/reduction temperature parity verification failed.')
    if STOP.exists():save_state(s,'paused_before_reduction');return
    previous=read(out/'workflow_state.json',{})
    if previous.get('status')!='completed':
        save_state(s,'reduction_running')
        action='resume' if previous else 'run'
        code=child(s,reduction/'run_reduction.py',action,'--output',out)
        save_state(s,reduction_exit_code=code)
    result=read(out/'workflow_state.json',{})
    summary=read(out/'summary.json',{})
    # Both binary and numeric subgroup analyses are required after reduction.
    rules_done=(out/'rules/summary.json').is_file()
    if result.get('status')=='completed' and not rules_done:
        if child(s,reduction/'run_reduction.py','rules','--output',out)!=0:raise RuntimeError('Rule analysis failed')
    if result.get('status')=='completed' and read(out/'rules/summary.json',{}).get('status')!='completed':
        raise RuntimeError('Reduction ended but subgroup analysis is incomplete; results remain saved.')
    done=result.get('status')=='completed'
    save_state(s,'completed' if done else 'paused_reduction',
        reduction_stop_reason=result.get('stop_reason'),accepted_reduction_steps=result.get('depth'))
    archive_reduction(s)
    print('Pipeline completed.' if done else 'Pipeline paused; use resume for this same run.',flush=True)

def start(workdir=None):
    if STATE.exists():raise RuntimeError('This folder already has a run. Use resume/status; START never overwrites or restarts it.')
    manifest=verify_package();run_id=dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    if workdir:
        w=Path(workdir).expanduser().resolve()
    elif os.name=='nt':
        local=os.environ.get('LOCALAPPDATA')
        if not local:raise RuntimeError('LOCALAPPDATA missing. Supply --workdir.')
        w=Path(local)/'HIBO_SingleSP_FullPipeline_v3'/run_id
    else:w=ROOT/'work'/run_id
    if w.exists() and any(w.iterdir()):raise RuntimeError('Work directory must be new/empty.')
    if w==ROOT or ROOT.is_relative_to(w):raise RuntimeError('Work directory must not contain the package.')
    w.mkdir(parents=True,exist_ok=True)
    for name in ('initial','reduction'):
        shutil.copytree(ROOT/name,w/name,dirs_exist_ok=False,ignore=shutil.ignore_patterns('__pycache__','.*'))
    s=dict(package_id=PACKAGE_ID,run_id=run_id,created_utc=now(),workdir=str(w),stage='prepared',
        package_manifest_sha256=manifest,old_samples_imported=False,input_csv_sha256=sha(ROOT/'initial/inputs/HIBO_MCMC_1min.csv'))
    save_state(s);STOP.unlink(missing_ok=True)
    print('Working results: '+str(w),flush=True)
    if child(s,w/'initial/run_case5.py','verify')!=0:raise RuntimeError('Initial runtime/data verification failed')
    proceed(s)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','resume','status','stop','verify','export'))
    parser.add_argument('--workdir',type=Path,help='Optional NEW work directory for start; defaults outside OneDrive on Windows.')
    args=parser.parse_args()
    if args.action=='status':
        s=read(STATE,{})
        if s:
            i,r,o=paths(s)
            s=dict(s,active=active(),initial_status=read(i/'results/single_sp_initial/status.json'),reduction_status=read(o/'status.json'))
        print(json.dumps(s or {'stage':'not_started'},indent=2,ensure_ascii=False));return
    if args.action=='stop':
        s=read(STATE)
        if not s or not active():print('No active pipeline; nothing stopped.');return
        request_stop(s);print('Graceful stop requested. Keep the running terminal open for checkpoint/export.');return
    if args.workdir and args.action!='start':parser.error('--workdir is only for start; resume uses the recorded path.')
    with RunLock():
        if args.action=='verify':
            print(json.dumps({'manifest_sha256':verify_package(),'inputs':read(ROOT/'initial/inputs/metadata.json')},indent=2));return
        if args.action=='start':start(args.workdir);return
        s=read(STATE)
        if not s:raise RuntimeError('No pipeline exists in this folder; use start.')
        if s.get('package_id')!=PACKAGE_ID:
            raise RuntimeError('This saved run belongs to another package version. Extract v3 to a new folder and use START; no old checkpoint was imported.')
        if verify_package()!=s['package_manifest_sha256']:raise RuntimeError('Package changed since start; cannot resume.')
        if args.action=='export':
            if not s.get('initial_archive'):archive_initial(s)
            if (paths(s)[2]/'workflow_state.json').exists():archive_reduction(s)
            return
        if s['stage']=='completed':print('Already completed. No calculation restarted.');return
        if s['stage']=='initial_not_converged':
            print('Initial MCMC reached a nonconverged terminal stage. Results are saved; reduction is blocked.');return
        STOP.unlink(missing_ok=True);proceed(s,resume=True)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Interrupted; saved stage/checkpoint files remain. Use status, then resume.',file=sys.stderr);sys.exit(130)
    except Exception as error:
        traceback.print_exc()
        # Keep the last durable stage. Do not write after losing/not acquiring the lock.
        sys.exit(1)
