"""Small-model diagnostic experiment: predict before seeing assertions/results.

One existing failing unittest is supplied, so this is not an end-to-end score.
This pilot supports a complete trace with one constructed stateful object and
primitive observations. Unsupported traces stop instead of guessing state.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

from pydantic import BaseModel,ConfigDict,Field

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class Prediction(BaseModel):
    model_config=ConfigDict(extra='forbid')
    step:int
    value:str|int|float|bool|None


class Predictions(BaseModel):
    model_config=ConfigDict(extra='forbid')
    returns:list[Prediction]=Field(max_length=32)
    limitation:str=Field(max_length=200)


class Edit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    old:str=Field(min_length=1,max_length=1800)
    new:str=Field(min_length=1,max_length=1800)
    reason:str=Field(max_length=250)


def method(source,line):
    candidates=[n for n in ast.walk(ast.parse(source)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
                and n.lineno<=line<=n.end_lineno]
    if not candidates:raise ValueError('Failure has no isolated function excerpt')
    node=min(candidates,key=lambda n:n.end_lineno-n.lineno)
    return node,'\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])


async def run(args):
    from quality_ladder import source_manifest
    from src.coding.harness import Workspace
    from src.coding.model import OpenRouterModel
    from src.coding.test_trace import trace_failed_test,target_test
    from src.coding.verifier import isolated_probe
    from src.eval.quality_cases import cases
    from src.runtime import hitl
    source=json.loads(Path(args.report).read_text())
    row=next(r for r in source['reports'] if r['case']==args.case)
    case=next(c for c in cases(source['seed']) if c.name==args.case)
    if case.domain!='software':raise ValueError('Requires a software development replay')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    root=output.parent/'quality'/('blind-trace-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
    for name in case.files:
        (root/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(row['workspace'])/name,root/name)
    directory=root/'.sea'/'replay';directory.mkdir(parents=True)
    ws=Workspace(root,directory,{'unit':[sys.executable,'-B','-m','unittest','-q']})
    model=OpenRouterModel({'model':source['model'],'provider_only':[source['provider_only']],
        'max_cost_usd':args.budget_usd,'max_input_usd_per_million':.17,'max_output_usd_per_million':.25,
        'max_output_tokens':384,'cloud_context_tokens':16384,'schema_hint':False})
    project=Path(__file__).resolve().parents[1]
    manifest=source_manifest(project)
    manifest[str(Path(__file__).relative_to(project))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report={'scope':__doc__,'source_report':args.report,'workspace':str(root),
        'model':source['model'],'provider':source['provider_only'],'budget_usd':args.budget_usd,
        'source_manifest':manifest,'requests':[]}
    started=time.monotonic();hitl.set_work_mode('auto')
    async def call(system,packet,schema):
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(packet,separators=(',',':'))}]
        report['requests'].append({'schema':schema.__name__,'messages':messages})
        return await model(messages,schema,'low')
    try:
        failed=await ws.execute('check',{'name':'unit'});report['initial_check']=failed
        if failed['ok']:raise ValueError('Requires an actual failed check')
        traced=await trace_failed_test(ws,'unit',failed);report['trace']=traced
        if not traced or not traced['ok']:raise ValueError('No usable actual-test trace')
        trace=traced['observation']
        target,path=target_test(ws.checks['unit'],failed['output'])
        rows=[e for e in trace['events'] if 'call' in e and str(e['at']).startswith(path+':')]
        if trace['omitted_events'] or sum(e['call']=='__init__' for e in rows)!=1:
            raise ValueError('Pilot requires an untruncated single-instance trace')
        inputs=[{'step':i,'call':e['call'],'args':e['in']} for i,e in enumerate(rows)]
        prediction_system=('Derive expected return values ONLY from the task and the ordered input calls. '
            'There is one constructed object. Simulate all mutations, including reads. Return predictions only for '
            'calls whose return value the specification defines; omit unspecified returns. No implementation, '
            'observed values or test expectations are supplied. State any ambiguity briefly.')
        prediction_input={'task':case.task,'operations':inputs}
        if args.prediction_report:
            prior=json.loads(Path(args.prediction_report).read_text())
            previous=prior['requests'][0]['messages']
            if (prior['model']!=source['model'] or prior['provider']!=source['provider_only']
                    or previous[0]['content']!=prediction_system
                    or json.loads(previous[-1]['content'])!=prediction_input):
                raise ValueError('Prior prediction task, operations or producer differs')
            predictions=Predictions.model_validate(prior['predictions'])
            report.update(prediction_origin=args.prediction_report,prior_attempt_usage=prior['usage'])
        else:predictions=await call(prediction_system,prediction_input,Predictions)
        report['predictions']=predictions.model_dump()
        seen=set();comparisons=[]
        for item in predictions.returns:
            if item.step in seen or not 0<=item.step<len(rows):raise ValueError('Invalid predicted step')
            seen.add(item.step)
            comparisons.append({'step':item.step,'call':rows[item.step]['call'],'args':rows[item.step]['in'],'expected':item.value,
                'actual':rows[item.step]['out'],'agrees':item.value==rows[item.step]['out']})
        report['comparisons']=comparisons
        if not comparisons or not all(c['agrees'] for c in comparisons):
            raise ValueError('Pilot cannot localize this case to the test; no edit authorized by its evidence')
        exception=next(e for e in reversed(trace['events']) if 'exception' in e and e['at'].startswith(path+':'))
        current=(root/path).read_text();node,excerpt=method(current,int(exception['at'].rsplit(':',1)[1]))
        original_methods={n.name for n in ast.walk(ast.parse(case.files[path])) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        if node.name in original_methods:raise ValueError('Do not edit an original test in this pilot')
        # Bind every predicted value to its actual inputs. A bare list of get()
        # values lets the repair model confuse distinct operations. Focus on the
        # assertion that actually failed; keep all predictions in the audit report.
        focused=[c for c in comparisons if rows[c['step']]['at']==exception['at']]
        if not focused:raise ValueError('No prediction for the failing assertion call')
        proposed=await call('Correct mistaken expectations in this NEW test using the task and the step-specific '
            'observed/predicted values. Preserve all operations and assertions; do not skip checks. Return one exact '
            'old/new edit within this function. Predictions are fallible hypotheses.',
            {'task':case.task,'function':excerpt,'failure':trace['failures'],'independent_predictions':focused},Edit)
        report['edit']=proposed.model_dump()
        if proposed.old not in excerpt:raise ValueError('Edit escaped the supplied newly added function')
        before=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith('assert') for n in ast.walk(node))
        edited=current.replace(proposed.old,proposed.new,1)
        changed_node=next(n for n in ast.walk(ast.parse(edited)) if isinstance(n,ast.FunctionDef) and n.name==node.name)
        after=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith('assert') for n in ast.walk(changed_node))
        if after<before:raise ValueError('Edit removed assertions')
        applied=await ws.execute('edit',{'path':path,'old':proposed.old,'new':proposed.new})
        report['applied']=applied
        if not applied['ok']:raise ValueError('Edit not applied')
        report['registered_check']=await ws.execute('check',{'name':'unit'})
        report['independent_check']=await isolated_probe(ws,case.acceptance)
        report['replay_checks_passed']=report['registered_check']['ok'] and report['independent_check']['ok']
    except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:700]
    finally:
        report.update(usage=model.usage(),seconds=round(time.monotonic()-started,2),
            source_unchanged=all(hashlib.sha256((project/n).read_bytes()).hexdigest()==h for n,h in manifest.items()))
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        ws._scratch.cleanup()
    print(json.dumps({k:report.get(k) for k in ['seconds','comparisons','replay_checks_passed','error','usage']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--report',required=True)
    parser.add_argument('--case',default='lru_cache')
    parser.add_argument('--prediction-report',help='Reuse an earlier blind prediction only for the same input and model/provider')
    parser.add_argument('--budget-usd',type=float,default=.00075)
    parser.add_argument('--output',default='data/eval/blind-trace-replay.json')
    args=parser.parse_args()
    if not 0<args.budget_usd<=.003:parser.error('budget must be in (0,.003]')
    if args.live:asyncio.run(asyncio.wait_for(run(args),60))
    else:print('No API calls. Use --live for the two-call diagnostic pilot.')
