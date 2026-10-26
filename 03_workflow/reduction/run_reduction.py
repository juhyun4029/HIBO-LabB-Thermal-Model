"""Command-line entry point for RC model reduction and rule discovery."""
from pathlib import Path
import argparse
import os
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'1')


def main():
    parser=argparse.ArgumentParser(description='HIBO Lab B: all-pairs R/C/alpha reduction v3, Steps 4–13')
    parser.add_argument('command',choices=['verify','plan','run','resume','smoke','status','stop','rules','export','reprocess'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--source',type=Path,help='Completed v2 result directory or ZIP for reprocess; defaults to bundled prior_results.')
    parser.add_argument('--reprocessed',action='store_true',help='Use the reprocessed output for status, rules or export.')
    parser.add_argument('--limit-trials',type=int,help='Pause after N newly completed MCMC attempts (including rescue); never adopt a partial wave.')
    parser.add_argument('--one-stage',action='store_true',help='Pause after evaluating all pairs and adopting one child.')
    args=parser.parse_args()
    from reduction.workflow import load_config,run,verify,plan,read_json,discover_all_rules,export_results,jsonable,default_output
    mode='reprocessed' if args.command=='reprocess' or args.reprocessed else 'smoke' if args.command=='smoke' else 'full'
    output=(args.output or default_output(ROOT,mode)).resolve()
    import json
    cfg=load_config(ROOT,args.config,smoke=args.command=='smoke')
    if args.source and args.command!='reprocess':parser.error('--source is only valid with reprocess')
    if args.reprocessed and args.command not in ('status','rules','export'):parser.error('--reprocessed is only valid with status, rules or export')
    if args.limit_trials is not None and args.limit_trials<1:parser.error('--limit-trials must be positive')
    if args.command=='verify':answer=verify(ROOT,ROOT/'validation/preflight_local.json')
    elif args.command=='plan':answer=plan(ROOT,cfg,args.output or default_output(ROOT,'plan'))
    elif args.command in ('run','resume','smoke'):
        state=run(ROOT,cfg,output,resume=args.command=='resume',
                  limit_trials=1 if args.command=='smoke' else args.limit_trials,one_stage=args.one_stage)
        answer={k:state.get(k) for k in ('status','stop_reason','depth','mode')}
        answer['output']=str(output)
    elif args.command=='status':
        p=output/'status.json';state=read_json(output/'workflow_state.json')
        answer={'output':str(output),'workflow':{k:state.get(k) for k in ('status','stop_reason','depth','mode')},
                'last_progress':read_json(p) if p.exists() else None}
    elif args.command=='stop':
        if not (output/'RUNNING.lock').exists():raise ValueError('No active run in this output directory.')
        (output/'STOP_REQUESTED').write_text('Please stop at next checkpoint.\n',encoding='utf-8')
        answer={'status':'stop_requested','output':str(output)}
    elif args.command=='rules':
        if (output/'RUNNING.lock').exists():raise RuntimeError('Wait for the run to stop before regenerating rules.')
        answer=discover_all_rules(read_json(output/'all_trials.json'),cfg,output/'rules')
    elif args.command=='reprocess':
        from reduction.reprocess import reprocess_completed
        answer=reprocess_completed(ROOT,cfg,args.source or ROOT/'prior_results',output)
    else:answer={'file':export_results(ROOT,output)}
    print(json.dumps(jsonable(answer),indent=2,ensure_ascii=False))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(f'ERROR: {error}',file=sys.stderr)
        raise SystemExit(1)
