"""Paired fixed-state controller comparison; not an end-to-end coding benchmark."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

CASES = [
    ('inspect','Locate the parser entry point and its callers; no edits yet.','plan',0,['inspect']),
    ('plan','Составь план миграции API и список рисков, пока ничего не меняй.','inspect',0,['plan']),
    ('implement','Execute the approved one-file change. The cause is known, the plan is recorded.','plan',0,['implement']),
    ('repair','The regression test reproduced an off-by-one in parser.py: expected 3, got 2. Fix the defect.','verify',2,['repair']),
    ('verify','The requested source change is complete. Run the existing registered tests before finishing.','implement',0,['verify']),
    ('review','Inspect this proposed patch for API compatibility and missing boundary cases; do not edit it.','implement',0,['review']),
    ('research','Найди в официальной документации неизвестный формат нового API. В репозитории этой информации нет.','reason',0,['research']),
    ('reason','Derive an invariant and correctness argument for the proposed algorithm; no source change is needed.','inspect',0,['reason']),
]


def packet(phase, failures):
    return {'phase':phase,'failures':failures,'plan':[],'summary':'','observations':[],
            'profiles':[],'human_steering':[],'allowed_tools':['read','tree','search','edit','write','check','math','experiment','web','fetch'],
            'writable_branches':True,'execution':'auto','trigger':'observation',
            'budget':{'nodes_left':3,'depth_left':2,'inference_slots':1}}


async def run(args):
    from src.coding.decision import DecisionRouter
    baseline=DecisionRouter(None,{'backend':'openrouter','model':args.worker_model,'max_calls':len(CASES),'max_cost_usd':.025})
    typed=DecisionRouter(None,{'backend':'decisions','model':args.decision_model,'max_calls':len(CASES),'max_cost_usd':.002,
        'planner':{'model':args.worker_model,'max_calls':2,'max_cost_usd':.01}})
    reports=[]
    for i,(name,task,phase,failures,expected) in enumerate(CASES):
        # Alternate ordering to reduce a fixed warm-up/order bias.
        engines=[('generative',baseline),('typed',typed)]
        if i%2: engines.reverse()
        for label,router in engines:
            started=time.monotonic();before=router.usage()
            try:
                result=await asyncio.wait_for(router.choose(task,packet(phase,failures)),45)
                row={'case':name,'variant':label,'expected_modes':expected,'result':result.model_dump() if result else None,
                     'passed':result is not None and result.mode in expected and result.operation!='continue'}
            except Exception as exc:
                row={'case':name,'variant':label,'passed':False,'error':type(exc).__name__+': '+str(exc)[:300]}
            row.update(seconds=round(time.monotonic()-started,3),cost=router.usage()['cost']-before['cost'],
                       tokens=router.usage()['tokens']-before['tokens'])
            reports.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    report={'scenario':'paired_controller_integration_not_benchmark','worker_model':args.worker_model,
            'decision_model':args.decision_model,'cases_sha256':hashlib.sha256(json.dumps(CASES).encode()).hexdigest(),
            'reports':reports,'usage':{'generative':baseline.usage(),'typed':typed.usage()}}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['usage']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--worker-model',default='qwen/qwen3.5-9b')
    parser.add_argument('--decision-model',default='typesafe/jev-1.13')
    parser.add_argument('--output',default='data/eval/controller-comparison.json')
    args=parser.parse_args()
    if args.live: asyncio.run(run(args))
    else: print('No API calls. Use --live for paired, bounded cloud calls.')
