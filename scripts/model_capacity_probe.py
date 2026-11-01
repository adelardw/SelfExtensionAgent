"""Direct-answer diagnostic, not an agent benchmark or evidence of task completion.

No tools are supplied to the model. Host-only finite answer checks measure whether
an answer is found; execution requirements and mathematical prose are NOT verified.
Compare this with complete agent runs rather than treating it as their replacement.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from pydantic import RootModel

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.coding.model import OpenRouterModel
from src.eval.quality_cases import cases,check_answer
from quality_ladder import source_manifest


class Candidate(RootModel[dict[str,Any]]):
    pass


async def run(args):
    project=Path(__file__).resolve().parents[1]
    manifest=source_manifest(project)
    manifest[str(Path(__file__).relative_to(project))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selected=[c for c in cases(args.seed) if c.domain!='software' and (not args.case or c.name in args.case)]
    model=OpenRouterModel({'model':args.model,'provider_only':[args.provider],
        'temperature':args.temperature,'top_p':args.top_p,
        'max_cost_usd':0,'max_input_usd_per_million':0,'max_output_usd_per_million':0,
        'max_output_tokens':8192,'cloud_context_tokens':32768,'provider_retries':1,
        'provider_retry_wait_seconds':3,'min_request_interval_seconds':3})
    report={'scope':__doc__,'seed':args.seed,'model':args.model,'provider':args.provider,'free_only':True,
            'sampling':dict(model.sampling),
            'source_manifest':manifest,'agent_completion_measured':False,'reports':[]}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    async def call(case,effort):
        messages=[{'role':'system','content':'Solve the task and return the exact JSON answer requested. '
            'You have no execution tools in this diagnostic. Do not claim to have executed or verified code. '
            'State that limitation in your explanation; provide the mathematical answer if you can derive it.'},
            {'role':'user','content':case.task}]
        before=model.usage();started=time.monotonic()
        row={'case':case.name,'reasoning':effort,'messages':messages}
        try:
            answer=await asyncio.wait_for(model(messages,Candidate,effort),90)
            row['answer']=answer.root
            try:row['finite_answer_correct']=check_answer(case,json.dumps(answer.root))
            except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError):row['finite_answer_correct']=False
        except Exception as exc:
            row.update(error=type(exc).__name__+': '+str(exc)[:400],finite_answer_correct=None)
        after=model.usage()
        row.update(seconds=round(time.monotonic()-started,2),
            usage={k:after.get(k,0)-before.get(k,0) for k in ('calls','tokens','tokens_in','tokens_out','reasoning_tokens','cost')})
        report['reports'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('messages','answer')}),flush=True)
    try:
        for effort in args.reasoning or ['low','medium']:
            for case in selected:await call(case,effort)
    finally:
        report.update(usage=model.usage(),source_unchanged=all(hashlib.sha256((project/n).read_bytes()).hexdigest()==h for n,h in manifest.items()))
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--model',default='cohere/north-mini-code:free')
    parser.add_argument('--provider',default='cohere')
    parser.add_argument('--seed',type=int,default=18149)
    parser.add_argument('--case',action='append',choices=[c.name for c in cases() if c.domain!='software'])
    parser.add_argument('--reasoning',action='append',choices=['low','medium','high'])
    parser.add_argument('--temperature',type=float,help='Omitted uses provider default')
    parser.add_argument('--top-p',type=float,help='Omitted uses provider default')
    parser.add_argument('--output',default='data/eval/direct-model-capacity.json')
    args=parser.parse_args()
    if args.live:asyncio.run(run(args))
    else:print('No API calls. Use --live for free-only direct-answer diagnostics.')
