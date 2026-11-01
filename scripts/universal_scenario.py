"""Opt-in cheap-cloud integration of the SAME engine on exact arithmetic and algorithm analysis.

These two small cases verify plumbing, not benchmark quality or general proof capability.
The checker is host-only and never passed to the model or included in its workspace.
"""
from __future__ import annotations

import argparse
import asyncio
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

CASES=[
    ('mathematics',
     'Compute exactly sum(k*(k+1)/((k+2)*(k+3)) for k=1..1000). '
     'Use an exact computation tool and explain a derivation. Do not create repository files. '
     'The final answer must be a JSON object with keys value (a reduced fraction p/q) and derivation (string).'),
    ('algorithms',
     'For unlimited coins of denominations [1,7,10], find the smallest positive integer amount '
     'at which largest-first greedy uses more coins than an optimal solution. '
     'Use an executable finite experiment, then describe a correct general algorithm and its invariant '
     'and complexity. Do not create repository files. Return the final answer as a JSON object with '
     'keys amount (integer), greedy (list of coin values), optimal (list of coin values), '
     'algorithm (string), correctness (string), complexity (string).')
]


def check(case,answer):
    value=json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
    if case=='mathematics':
        expected=sum((Fraction(k*(k+1),(k+2)*(k+3)) for k in range(1,1001)),Fraction())
        return Fraction(value['value'])==expected and bool(value['derivation'])
    best=[0]+[1000]*100
    first=None
    for amount in range(1,101):
        best[amount]=min(1+best[amount-c] for c in (1,7,10) if c<=amount)
        left=amount;greedy=[]
        for coin in (10,7,1):
            greedy.extend([coin]*(left//coin));left%=coin
        if len(greedy)>best[amount]:first=amount;break
    return (value['amount']==first and value['greedy']==greedy
            and all(c in (1,7,10) for c in value['optimal'])
            and sum(value['optimal'])==first and len(value['optimal'])==best[first]
            and all(value.get(k) for k in ('algorithm','correctness','complexity')))


async def evaluate(args):
    from src.harness import Harness
    from src.config.config_paths import load_base_config
    from src.runtime import hitl,run_context
    reports=[]
    for name,task in CASES:
        root=Path(args.work_dir)/(name+'-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
        config=dict(load_base_config().get('coding',{}))
        config.update(backend='openrouter',model=args.model,checks={},max_turns=12,
                      max_cost_usd=.025,max_total_tokens=60000,allow_web=False,learn=False)
        config['orchestration']={**config.get('orchestration',{}),'same_model':True,
                                 'auto_domains':True,'observer':False}
        config['orchestration']['critic']={**config['orchestration'].get('critic',{}),
                                           'enabled':True,'max_calls':4}
        started=time.monotonic();agent=Harness(root,config=config)
        with run_context.request_scope('universal-'+name,'integration'):
            hitl.set_work_mode('auto')
            try:
                result=await asyncio.wait_for(agent.run(task,reasoning='low',session_id=name),timeout=100)
            except Exception as exc:
                result={'status':'error','error':type(exc).__name__+': '+str(exc),**agent.model.usage()}
        try: passed=check(name,result.get('final_answer',''))
        except (ValueError,KeyError,TypeError,ZeroDivisionError): passed=False
        tools=result.get('tools_used',[])
        report={'case':name,'workspace':str(root),'seconds':round(time.monotonic()-started,2),
                'independent_answer_passed':passed,
                'required_tool_used':('math' if name=='mathematics' else 'experiment') in tools,
                'source_unchanged':not any(p for p in root.iterdir() if p.name!='.sea'),
                'proof_validated':False,'result':result}
        reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k!='result'}|{
            'status':result['status'],'cost':result.get('cost'),'tools':tools},ensure_ascii=False),flush=True)
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        code=Path(__file__).resolve().parents[1]/'src'/'coding'
        output.write_text(json.dumps({'scenario':'two_domain_integration_not_benchmark',
            'harness_id':hashlib.sha256(b''.join(p.read_bytes() for p in sorted(code.glob('*.py')))).hexdigest()[:16],
            'same_model_all_roles':True,'reports':reports},ensure_ascii=False,indent=2))
    return 0 if all(r['independent_answer_passed'] and r['required_tool_used'] and r['source_unchanged']
                    and r['result']['status']=='done' for r in reports) else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--model',default='google/gemini-3.1-flash-lite')
    parser.add_argument('--work-dir',default='data/eval/universal')
    parser.add_argument('--output',default='data/eval/universal-scenario.json')
    args=parser.parse_args()
    if not args.live:
        print('No API calls. Use --live for two bounded cheap-cloud integrations.');return 0
    return asyncio.run(evaluate(args))


if __name__=='__main__':raise SystemExit(main())
