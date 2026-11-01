"""Bounded replay of a paused development failure, not an end-to-end benchmark.

Uses the same cheap cloud model for an executable diagnosis and up to two focused
repair actions. Hidden acceptance stays outside the model's disposable workspace.
No API calls without --live; source reports and their workspaces are never edited.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


async def run(args):
    from quality_ladder import source_manifest
    from src.coding.control import SessionControl
    from src.coding.critic import ContextCritic
    from src.coding.failure_diagnosis import diagnose_failure,diagnostic_packet
    from src.coding.harness import Workspace,Turn
    from src.coding.minds import MindScheduler
    from src.coding.model import OpenRouterModel
    from src.coding.orchestration import scoped_schema
    from src.coding.prompts import system_prompt,action_policy
    from src.coding.supervisor import Supervisor
    from src.coding.verifier import isolated_probe
    from src.eval.quality_cases import cases
    from src.runtime import hitl

    source=json.loads(Path(args.report).read_text())
    row=next(r for r in source['reports'] if r['case']==args.case)
    case=next(c for c in cases(source['seed']) if c.name==args.case)
    if case.domain!='software':raise ValueError('Replay currently supports software cases with isolated checks')
    project=Path(__file__).resolve().parents[1]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    root=output.parent/'quality'/('failure-replay-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
    for name in case.files:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(row['workspace'])/name,target)
    directory=root/'.sea'/'coding'/'replay';directory.mkdir(parents=True)
    ws=Workspace(root,directory,{'unit':['python3','-m','unittest','discover','-q']})
    ws.changed.update(case.files)
    control=SessionControl();hitl.set_work_mode('auto')
    model_name=args.model or source['model'];provider=args.provider or source['provider_only']
    model=OpenRouterModel({'model':model_name,'provider_only':[provider],
        'max_cost_usd':0 if args.free_only else args.budget_usd,
        'max_input_usd_per_million':0 if args.free_only else .17,
        'max_output_usd_per_million':0 if args.free_only else .25,
        'max_output_tokens':args.max_output_tokens,'cloud_context_tokens':32768,
        'min_request_interval_seconds':args.request_interval})
    supervisor=Supervisor(directory,control,model,False);control.supervisor=supervisor
    model.on_event=supervisor.model_event
    node=MindScheduler(control,supervisor,{},ws).nodes['main'];node.task=case.task
    node.memory.files={name:ws.file_digest(name) for name in case.files}
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'max_calls':8})
    manifest=source_manifest(project)
    manifest[str(Path(__file__).relative_to(project))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with zipfile.ZipFile(output.with_suffix('.sources.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:archive.write(project/name,name)
    report={'scope':'Focused failure replay; preexisting source/test work is supplied, so this is not an end-to-end task score.',
        'source_report':args.report,'case':case.name,'workspace':str(root),'model':model_name,
        'provider_only':provider,'budget_usd':0 if args.free_only else args.budget_usd,
        'free_only':args.free_only,'max_output_tokens':args.max_output_tokens,'request_interval_seconds':args.request_interval,
        'max_actions':args.actions,'source_manifest':manifest,
        'source_id':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),'actions':[]}
    started=time.monotonic()
    try:
        failed=await ws.execute('check',{'name':'unit'});eid=ws.record(failed)
        report['initial_check']=failed
        if failed['ok']:raise ValueError('Replay requires a currently failing registered check')
        diagnosis=await diagnose_failure(critic,node,ws,'unit',eid)
        report['diagnosis']=diagnosis
        for _ in range(args.actions):
            schema=scoped_schema(Turn,{'edit','read','experiment'},can_delegate=False,can_plan=False)
            payload={'task':case.task,'phase':'repair','checks':ws.checks,'failed_check':failed,
                'current_source':{name:await ws.source_text(name,5000) for name in case.files},
                'diagnosis':diagnostic_packet(diagnosis),'recent':report['actions'][-1:],
                'next_action_hint':'Correct the actual defect using the task and observed behavior. Your new tests may be wrong. Preserve original assertions. Do not repeat the existing plan.'}
            with supervisor.scope('main'):
                turn=await control.call(lambda:model([
                    {'role':'system','content':system_prompt({'edit','read','experiment'},'Repair the observed failure.',domains=['software'],focused=True)+action_policy(schema)},
                    {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],schema,'low'))
            if turn.action!='tool':
                report['actions'].append({'turn':turn.model_dump()});break
            result=await ws.execute(turn.tool,turn.args)
            report['actions'].append({'turn':turn.model_dump(),'result':result})
            if result['ok'] and turn.tool=='edit':
                checked=await ws.execute('check',{'name':'unit'})
                report['registered_check']=checked
                if checked['ok']:break
                failed=checked
                # A diagnosis of old source must not anchor the next repair.
                report['diagnosis_after_edit']='Historical only; use current source and the latest failed check'
                diagnosis=None
        report['independent_check']=await isolated_probe(ws,case.acceptance)
        report['replay_checks_passed']=bool(report.get('registered_check',{}).get('ok') and report['independent_check']['ok'])
    except asyncio.CancelledError:
        report['error']='Cancelled or timed out; replay remains incomplete'
        raise
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc)[:700]
    finally:
        report.update(usage=model.usage(),seconds=round(time.monotonic()-started,2),
            source_unchanged=all(hashlib.sha256((project/n).read_bytes()).hexdigest()==h for n,h in manifest.items()))
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        ws._scratch.cleanup()
    print(json.dumps({k:report.get(k) for k in ('seconds','replay_checks_passed','error','usage')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--report',required=True)
    parser.add_argument('--case',default='lru_cache')
    parser.add_argument('--model',help='Explicit override; all replay roles use this same model')
    parser.add_argument('--provider',help='Explicit provider override; no fallback')
    parser.add_argument('--budget-usd',type=float,default=.003)
    parser.add_argument('--free-only',action='store_true',help='Require zero input/output/request prices and forbid paid routing')
    parser.add_argument('--max-output-tokens',type=int,default=2000)
    parser.add_argument('--request-interval',type=float,default=0)
    parser.add_argument('--actions',type=int,choices=[0,1,2],default=2,help='Zero tests diagnosis only, not repair success')
    parser.add_argument('--timeout',type=int,default=90)
    parser.add_argument('--output',default='data/eval/failure-replay.json')
    args=parser.parse_args()
    if not 0<args.budget_usd<=.02:parser.error('budget must be in (0,.02]')
    if not 256<=args.max_output_tokens<=8192:parser.error('max-output-tokens must be in [256,8192]')
    if not 0<=args.request_interval<=30:parser.error('request-interval must be in [0,30]')
    if not 10<=args.timeout<=180:parser.error('timeout must be in [10,180]')
    if args.live:asyncio.run(asyncio.wait_for(run(args),args.timeout))
    else:print('No API calls. Use --live for the bounded failure replay.')
