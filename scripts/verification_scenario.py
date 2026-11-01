"""Opt-in cheap-cloud negative/positive control for the active verifier, not a benchmark.

The producer is a fixed fixture: this tests independent review, not autonomous coding.
An incorrect greedy implementation passes its old tests. The reviewer must discover
a counterexample; a separately supplied DP implementation must then pass review.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BAD = '''def minimum_coins(amount):
    count = 0
    for coin in (10, 7, 1):
        count += amount // coin
        amount %= coin
    return count
'''
GOOD = '''def minimum_coins(amount):
    best = [0] + [amount + 1] * amount
    for total in range(1, amount + 1):
        best[total] = min(best[total - c] + 1 for c in (1, 7, 10) if c <= total)
    return best[amount]
'''


async def run(args):
    from src.coding.harness import Workspace
    from src.coding.model import OpenRouterModel
    from src.coding.critic import ContextCritic
    from src.coding.control import SessionControl
    from src.coding.supervisor import Supervisor
    from src.coding.minds import MindScheduler
    from src.runtime import hitl, run_context
    root = Path(tempfile.mkdtemp(prefix='sea-verifier-control-'))
    directory = root/'.sea'; directory.mkdir()
    ws = Workspace(root,directory,{})
    model = OpenRouterModel({'model':'google/gemini-3.1-flash-lite','max_cost_usd':.025})
    control = SessionControl()
    supervisor = Supervisor(directory,control,model,False)
    node = MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task = ('Review app.py: minimum_coins(amount) must return the minimum number of coins '
                 'for every nonnegative integer amount, with unlimited denominations 1, 7, 10. '
                 'Use independent executable checks; do not change source.')
    critic = ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'max_calls':6,'max_probes':2})
    reports=[]
    with run_context.request_scope('active-verifier-control','integration'):
        hitl.set_work_mode('auto')
        for label,code in [('incorrect',BAD),('correct',GOOD)]:
            (root/'app.py').write_text(code)
            ws.revision+=1
            node.memory.files={'app.py':ws.file_digest('app.py')}
            node.memory.summary='Implementation complete; tests at 0, 1, 7, 10 pass.'
            node.memory.evidence=[ws.record({'ok':True,'output':'Prior tests: 0→0, 1→1, 7→1, 10→1'})]
            started=time.monotonic()
            result=await asyncio.wait_for(critic.assess(node,ws,'candidate completion'),70)
            reports.append({'fixture':label,'ready':result.ready,'seconds':round(time.monotonic()-started,2),
                            'assessment':result.model_dump(),'graph':ws.obligations().packet(),
                            'source_unchanged':(root/'app.py').read_text()==code})
            print(json.dumps({'fixture':label,'ready':result.ready,'issues':result.issues,'usage':model.usage()},ensure_ascii=False),flush=True)
    report={'scenario':'active_verifier_negative_positive_control_not_benchmark','workspace':str(root),
            'producer':'fixed fixtures, no LLM','reviewer':model.name,'usage':model.usage(),'reports':reports}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if not reports[0]['ready'] and reports[1]['ready'] and all(r['source_unchanged'] for r in reports) else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',default='data/eval/active-verifier-control.json')
    args=parser.parse_args()
    if not args.live:
        print('No API calls. Use --live for bounded cheap-cloud verification.')
    else:
        raise SystemExit(asyncio.run(run(args)))
