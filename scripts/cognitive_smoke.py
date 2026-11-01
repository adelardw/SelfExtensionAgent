"""Opt-in cheap-cloud smoke of routing, isolated handoffs and a live profile switch.

uv run python scripts/cognitive_smoke.py --live
No Docker, local inference or benchmark/oracle data is used.
"""
import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from src.coding.control import SessionControl
from src.coding.harness import CodingHarness, Workspace
from src.coding.model import OpenRouterModel
from src.coding.orchestration import Orchestrator
from src.coding.supervisor import Supervisor


async def run(output):
    with tempfile.TemporaryDirectory(prefix='sea-cognition-') as temp:
        root=Path(temp);directory=root/'.sea'/'coding'/'smoke';directory.mkdir(parents=True)
        (root/'parser.py').write_text('def parse_port(value):\n    return int(value)\n')
        (root/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\nproject(example)\nset(CMAKE_CXX_STANDARD 17)\n')
        model=OpenRouterModel({'model':'google/gemini-3.1-flash-lite','max_cost_usd':.03})
        control=SessionControl();control.bind(directory/'steering.jsonl')
        control.post('Preserve existing public interfaces.',interrupt=False)
        ws=Workspace(root,directory,{})
        supervisor=Supervisor(directory,control,model,False);control.supervisor=supervisor
        harness=CodingHarness(root,model=model,config={'max_total_tokens':40000});harness.control=control
        o=Orchestrator(harness,ws,supervisor,{'local_slots':2,'max_nodes':7,'max_depth':2,
            'max_branches':2,'mind_turns':5,'decision':{'backend':'openrouter',
            'model':'google/gemini-3.1-flash-lite','max_calls':2,'max_cost_usd':.003}})
        o.update_main('Review parser and build configuration','implement','API changes must be avoided',
                      ['Inspect parser bounds','Inspect build settings'],[],{})
        started=asyncio.Event()
        original=model.on_event
        def observe(kind,data):
            original(kind,data)
            if kind=='request': started.set()
        model.on_event=observe
        task=asyncio.create_task(o.delegate('Independently review parser.py input boundaries and CMakeLists.txt build flags; return separate evidence-backed findings.','review'))
        await asyncio.wait_for(started.wait(),40)
        # Switch a live branch without a new agent id or discarded working memory.
        node=next(n for key,n in o.scheduler.nodes.items() if key!='main')
        before=id(node.memory)
        o.scheduler.switch(node.id,'review',['check-edge-cases'])
        results=await asyncio.wait_for(task,90)
        report={'evaluation':'cognitive_smoke_not_benchmark','results':results,
            'mode_switch':{'id':node.id,'same_memory':id(node.memory)==before,'generation':node.generation},
            'minds':o.scheduler.report(),'worker_usage':model.usage(),
            'router_usage':{'calls':o.router.calls,'tokens':o.router.tokens,'cost':o.router.cost}}
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='minds'},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',default='data/eval/cognitive-cloud.json')
    args=parser.parse_args()
    if not args.live:
        print('No API calls. Pass --live to use the bounded cheap-cloud profile.');return
    asyncio.run(run(args.output))

if __name__=='__main__': main()
