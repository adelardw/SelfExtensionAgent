"""Explicit live Codex semantic-routing experiment; no production dispatch or effects."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, ConfigDict, Field
from src.eval.refinement import Proposal, ROUTING_PROMPT, ROUTING_PROMPT_V2, ShadowRouter, fingerprint
from src.eval.refinement_cases import TASKS, cases, expected, score
from codex_behavior_eval import CodexInference


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    case_id: str
    proposal: Proposal


class Batch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decisions: list[Decision] = Field(min_length=1,max_length=6)


async def evaluate(directory, policy='v1', batch_size=3):
    if not 1 <= batch_size <= 6: raise ValueError('batch size must be 1..6')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'calls').exists(): raise ValueError('Use a fresh output directory; never overwrite trials')
    shared={'calls':0,'tokens':0}
    model=CodexInference(directory/'calls',shared,max_calls=8)
    model.config['max_total_tokens']=180000
    prompt=ROUTING_PROMPT if policy=='v1' else ROUTING_PROMPT_V2
    frozen=cases()
    (directory/'cases.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
    (directory/'protocol.json').write_text(json.dumps({
        'prompt':prompt,'policy':policy,'cases_sha256':fingerprint(frozen),
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'src/eval/refinement.py',
             Path(__file__).resolve().parents[1]/'src/eval/refinement_cases.py')},
        'limits':{'calls':8,'tokens':180000},'batch_size':batch_size,
        'gold_sent_to_model':False,'backend':'authenticated Codex CLI default; no API key extracted',
        'production_integration':False},ensure_ascii=False,indent=2))
    results=[]

    async def run(batch, routers=None):
        cards=[{'case_id':c['id'],'tasks':c['tasks'],'event':c['event']} for c in batch]
        start=time.monotonic()
        try:
            response=await model([{'role':'system','content':prompt},
                {'role':'user','content':json.dumps({'independent_cases':cards},ensure_ascii=False)}],Batch,'low')
            decisions={r.case_id:r.proposal for r in response.decisions}
            if len(decisions)!=len(response.decisions) or set(decisions)!={c['id'] for c in batch}:
                raise ValueError('response must cover each case exactly once')
        except Exception as exc:
            for case in batch:results.append({'id':case['id'],'exact':False,'error':type(exc).__name__+': '+str(exc)[:300]})
            (directory/'partial.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
            raise
        for case in batch:
            proposal=decisions[case['id']]
            router=routers[case['id']] if routers else ShadowRouter(case['tasks'])
            snapshot=router.snapshot()
            row={'id':case['id'],**score(proposal,case['expected']),
                 'proposal':proposal.model_dump(),'batch_wall_seconds':round(time.monotonic()-start,3)}
            try:
                result=router.apply(case['event'],proposal,snapshot)
                after=copy.deepcopy(router.tasks)
                # Retry is permitted even with the original now-stale snapshot.
                repeated=router.apply(case['event'],proposal,snapshot)
                row.update(accepted=True,replay_deduplicated=repeated['duplicate'] and router.tasks==after,
                           result=result,state=after,pending=copy.deepcopy(router.pending))
                row['untouched_constraints_retained']=all(
                    router.tasks[tid]['requirements'].get(rid)==value
                    for tid,task in case['tasks'].items() for rid,value in task['requirements'].items()
                    if not any(c['task_id']==tid and c['requirement_id']==rid and c['operation'] in {'revise','retract'}
                               for c in case['expected']))
            except ValueError as exc:
                row.update(accepted=False,exact=False,operation_match=False,error=str(exc),atomic_rejection=router.snapshot()==snapshot)
            results.append(row)
            print(json.dumps({'case':row['id'],'exact':row['exact'],'accepted':row['accepted']}),flush=True)
        (directory/'partial.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

    for offset in range(0,len(frozen),batch_size):await run(frozen[offset:offset+batch_size])
    # Two actual successive model decisions: the second sees runtime state from
    # the first, never a gold reconstruction or hidden correction.
    router=ShadowRouter(TASKS)
    sequence=[('sequence-1','В A замени JSON API v1 на JSON API v2.',expected('revise','A','r-api',contains=['v2'])),
              ('sequence-2','Теперь для A измени предел p95 на 80 мс. Прежние требования к API и Python сохраняются.',
               expected('revise','A','r-latency',contains=['80']))]
    for name,text,gold in sequence:
        case={'id':name,'tasks':copy.deepcopy(router.tasks),'event':{'id':name,'text':text,'origin':'user','task_hint':'A'},'expected':[gold]}
        await run([case],{name:router})
    report={'cases':len(results),'exact':sum(r['exact'] for r in results),
        'operation_match':sum(r.get('operation_match',False) for r in results),
        'accepted':sum(r.get('accepted',False) for r in results),
        'untouched_constraints_retained':sum(r.get('untouched_constraints_retained',False) for r in results),
        'replay_deduplicated':sum(r.get('replay_deduplicated',False) for r in results),
        'calls':shared['calls'],'usage':{k:v for k,v in model.usage().items() if not k.startswith('cost')},
        'cost_usd':None,'cost_note':'ChatGPT-authenticated CLI; token counts are available, dollar cost is unknown.',
        'limitations':['hand-labelled synthetic diagnostic cases, one trial each',f'up to {batch_size} independent cases per model request',
            'strong CLI default model, not the configured Flash/9B','shadow state transitions, no worker launch or real side effects'],
        'results':results}
    (directory/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'results','limitations'}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('data/eval/refinement-20261106/live'))
    parser.add_argument('--policy',choices=['v1','v2'],default='v1')
    parser.add_argument('--batch-size',type=int,choices=range(1,7),default=3)
    args=parser.parse_args()
    if args.live:asyncio.run(evaluate(args.output.resolve(),args.policy,args.batch_size))
    else: print(json.dumps({'cases':len(cases())+2,'max_calls':8,'live':False}))
