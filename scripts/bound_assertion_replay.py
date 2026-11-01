"""Blind scalar expectations bound mechanically into a NEW test, diagnostic only.

Replays all operations in a supported straight-line test without stopping at its
first wrong assertion. This modified experiment is not the original test run.
No API calls without --live. No strong model, no runtime default activation.
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
from pydantic import BaseModel,ConfigDict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class Expected(BaseModel):
    model_config=ConfigDict(extra='forbid')
    values:list[str|int|float|bool|None]


class Reference(BaseModel):
    model_config=ConfigDict(extra='forbid')
    code:str


async def run(args):
    from quality_ladder import source_manifest
    from src.coding.assertion_repair import prepare_assertion_plan
    from src.coding.harness import Workspace
    from src.coding.model import OpenRouterModel
    from src.coding.test_trace import target_test
    from src.coding.verifier import isolated_probe
    from src.eval.quality_cases import cases
    from src.runtime import hitl
    source=json.loads(Path(args.report).read_text());row=next(r for r in source['reports'] if r['case']==args.case)
    case=next(c for c in cases(source['seed']) if c.name==args.case)
    if case.domain!='software':raise ValueError('Requires a development failure')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    root=output.parent/'quality'/('bound-assertion-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
    for name in case.files:
        (root/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(row['workspace'])/name,root/name)
    directory=root/'.sea'/'replay';directory.mkdir(parents=True)
    ws=Workspace(root,directory,{'unit':[sys.executable,'-B','-m','unittest','-q']})
    model=OpenRouterModel({'model':source['model'],'provider_only':[source['provider_only']],
        'max_cost_usd':args.budget_usd,'max_input_usd_per_million':.17,'max_output_usd_per_million':.25,
        'max_output_tokens':args.max_output_tokens,'cloud_context_tokens':8192,'schema_hint':False})
    project=Path(__file__).resolve().parents[1];manifest=source_manifest(project)
    manifest[str(Path(__file__).relative_to(project))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report={'scope':__doc__,'source_report':args.report,'workspace':str(root),'model':model.name,
        'provider':source['provider_only'],'budget_usd':args.budget_usd,'source_manifest':manifest}
    started=time.monotonic();hitl.set_work_mode('auto')
    try:
        failed=await ws.execute('check',{'name':'unit'});report['initial_check']=failed
        if failed['ok']:raise ValueError('Expected failed check')
        target=target_test(ws.checks['unit'],failed['output'])
        if not target:raise ValueError('No supported failed unittest')
        name,path=target;cls,method=name.split('.')[-2:]
        original=(root/path).read_text()
        plan=prepare_assertion_plan(original,case.files[path],cls,method)
        if not await ws.file_exists(plan.module.replace('.','/')+'.py'):raise ValueError('Implementation must be a project file')
        result=await isolated_probe(ws,plan.observation_program(),preserve_source=True)
        report['observation']=result
        if not result['ok'] or result.get('stale'):raise ValueError('Unusable observations')
        values=[json.loads(l.removeprefix('SEA_ASSERTION_VALUES ')) for l in result['output'].splitlines() if l.startswith('SEA_ASSERTION_VALUES ')]
        if len(values)!=1:raise ValueError('Ambiguous observations')
        messages=([{'role':'system','content':'Fill EXPECTED_1..N from the task and execution order. Return values in that order. '
            'Reads may mutate state. Original expectations and actual outputs were withheld.'},
            {'role':'user','content':json.dumps({'task':case.task,'test':plan.masked_test()},separators=(',',':'))}]
            if args.input_format=='code' else [
            {'role':'system','content':'Simulate the task specification for these ordered calls on one object. '
            'Return expected values only for predict_steps (zero-based), in that order. No implementation or test expectations are supplied.'},
            {'role':'user','content':json.dumps({'task':case.task,**plan.blind_input()},separators=(',',':'))}])
        report.update(input_format=args.input_format,oracle=args.oracle)
        if args.oracle=='program':
            messages=[{'role':'system','content':'Independent Python reference. Export '+plan.constructor+'; no tests.'},
                      {'role':'user','content':case.task}]
        report['request']=messages
        if args.oracle=='program':
            reference=await model(messages,Reference,'low');report['reference']=reference.model_dump()
            # The oracle receives no candidate code or expectations, and runs in
            # an EMPTY workspace. The operation sequence is applied afterwards.
            oracle_root=directory/'oracle';oracle_root.mkdir()
            oracle_dir=oracle_root/'.sea';oracle_dir.mkdir()
            oracle_ws=Workspace(oracle_root,oracle_dir,{})
            try:executed=await oracle_ws.command([sys.executable,'-c',plan.observation_program(reference.code)],timeout=15)
            finally:oracle_ws._scratch.cleanup()
            report['oracle_execution']=executed
            if not executed['ok']:raise ValueError('Reference execution failed')
            outputs=[json.loads(l.removeprefix('SEA_ASSERTION_VALUES ')) for l in executed['output'].splitlines() if l.startswith('SEA_ASSERTION_VALUES ')]
            if len(outputs)!=1 or len(outputs[0])!=len(plan.operations):raise ValueError('Incomplete reference values')
            predictions=Expected(values=[outputs[0][a['step']] for a in plan.assertions])
        else:predictions=await model(messages,Expected,'low')
        report['predictions']=predictions.model_dump()
        if any(ws.file_digest(name)!=value for name,value in result['source_hashes'].items()):
            raise ValueError('Source changed after observations; predictions cannot bind this revision')
        proposal=plan.proposal(predictions.values,values[0]);report['proposed_test_source']=proposal
        applied=await ws.execute('edit',{'path':path,'old':original,'new':proposal});report['applied']=applied
        if not applied['ok']:raise ValueError('Could not apply candidate to disposable replay')
        report['registered_check']=await ws.execute('check',{'name':'unit'})
        report['independent_check']=await isolated_probe(ws,case.acceptance)
        report['replay_checks_passed']=report['registered_check']['ok'] and report['independent_check']['ok']
    except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:700]
    finally:
        report.update(usage=model.usage(),seconds=round(time.monotonic()-started,2),
            source_unchanged=all(hashlib.sha256((project/n).read_bytes()).hexdigest()==h for n,h in manifest.items()))
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');ws._scratch.cleanup()
    print(json.dumps({k:report.get(k) for k in ['seconds','predictions','replay_checks_passed','error','usage']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true');parser.add_argument('--report',required=True)
    parser.add_argument('--case',default='lru_cache');parser.add_argument('--budget-usd',type=float,default=.00038)
    parser.add_argument('--input-format',choices=['code','operations'],default='code')
    parser.add_argument('--max-output-tokens',type=int,default=128)
    parser.add_argument('--oracle',choices=['values','program'],default='values')
    parser.add_argument('--output',default='data/eval/bound-assertion-replay.json');args=parser.parse_args()
    if not 0<args.budget_usd<=.003:parser.error('budget must be in (0,.003]')
    if not 32<=args.max_output_tokens<=512:parser.error('max-output-tokens must be in [32,512]')
    if args.live:asyncio.run(asyncio.wait_for(run(args),45))
    else:print('No API calls. Use --live for the one-call diagnostic pilot.')
