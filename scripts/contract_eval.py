"""Focused contract extraction diagnostics; not an end-to-end quality score.

Uses the production extractor and the same explicitly selected free cloud model
for every case. No API calls without --live. Expectations never enter prompts.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

CASES=[
 {'name':'json_math','task':'Compute the exact sum of reciprocals of integers 2 through 8. Explain the derivation and verify with exact executable arithmetic. Do not create project files. Return JSON with value (reduced fraction p/q) and derivation.',
  'mechanical':[],'keys':['value','derivation']},
 {'name':'algorithm_probe','task':'Explain an algorithm for detecting a directed cycle, its invariant and complexity. Verify it on finite examples using a scratch experiment. Do not create project files.',
  'mechanical':[],'keys':[]},
 {'name':'existing_tests_only','task':'Inspect parser.py without editing it. Run the existing registered unit check and report any failures. Do not add or modify tests.',
  'mechanical':['registered_checks'],'keys':[]},
 {'name':'new_tests','task':'Fix parser.parse_record while preserving its public signature. Add regression tests. Run the registered unit check.',
  'mechanical':['new_or_updated_tests','registered_checks'],'keys':[]},
 {'name':'readonly_worker','task':'Delegate an independent read-only review of parser.py. Implement the fix in the parent. Add regression tests and run the registered unit check.',
  'mechanical':['new_or_updated_tests','registered_checks'],'keys':[]},
 {'name':'worker_changes','task':'Delegate an implementation of a parser fix to a worker. Inspect and integrate the worker patch into the parent workspace. Run the registered unit check.',
  'mechanical':['adopted_worker_changes','registered_checks'],'keys':[]},
 {'name':'named_artifact','task':'Write notes/analysis.md containing the stated assumptions and an explanation of the existing algorithm. Do not modify source code or tests.',
  'mechanical':['artifact_exists'],'keys':[]},
 {'name':'russian_math','task':'Вычисли точно сумму квадратов чисел от 1 до 17. Объясни формулу и проверь результат вычислением. Файлы проекта не создавай. Верни JSON с ключами value и derivation.',
  'mechanical':[],'keys':['value','derivation']},
 {'name':'mixed_json_software','task':'Fix parser.parse_record while preserving its public signature. Add regression tests and run the registered unit check. Return JSON with status and tests.',
  'mechanical':['new_or_updated_tests','registered_checks'],'keys':['status','tests']},
]


def evaluate(case,contract):
    observed={r['validation'] for r in contract['requirements'] if r['validation']!='model_evidence'}
    expected=set(case['mechanical'])
    issues=[]
    if observed!=expected:issues.append({'mechanical_expected':sorted(expected),'mechanical_observed':sorted(observed)})
    if set(contract['response_keys'])!=set(case['keys']):issues.append({'answer_keys_expected':case['keys'],'answer_keys_observed':contract['response_keys']})
    if contract['response_format']!=('json' if case['keys'] else 'text'):issues.append({'response_format':contract['response_format']})
    # Category accuracy alone is not full extraction completeness or entailment.
    return {'passed':not issues,'issues':issues}


async def run(args):
    from src.coding.model import OpenRouterModel,ProviderUnavailable
    from src.coding.critic import ContextCritic
    from src.coding.control import SessionControl
    from src.coding.supervisor import Supervisor
    from scripts.quality_ladder import source_manifest
    root=Path(__file__).resolve().parents[1]
    def manifest():
        return {**source_manifest(root),'scripts/contract_eval.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    sources=manifest()
    config={'model':args.model,'provider_only':[args.provider],'max_output_tokens':2048,
        'cloud_context_tokens':32768,'max_cost_usd':0,'max_input_usd_per_million':0,
        'max_output_usd_per_million':0,'provider_retries':2,'provider_retry_wait_seconds':5,
        'min_request_interval_seconds':3,'structured_output':'auto'}
    model=OpenRouterModel(config)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    directory=output.parent/'contract-runs'/uuid.uuid4().hex[:12];directory.mkdir(parents=True)
    report={'scope':__doc__,'model':args.model,'provider':args.provider,'configuration':config,
        'source_manifest':sources,'source_id':hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
        'expectations_withheld':True,'reports':[]}
    def save():
        report['usage']=model.usage();report['source_unchanged']=sources==manifest()
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for case in CASES:
        if args.case and case['name'] not in args.case:continue
        if model.usage()['tokens']>=30000:
            report['stopped']='Total token diagnostic allowance reached';break
        control=SessionControl();supervisor=Supervisor(directory,control,model,False)
        critic=ContextCritic(model,supervisor,control,{'enabled':True,'max_calls':2,'answer_verification':True})
        node=SimpleNamespace(id=case['name'],task=case['task'])
        before=model.usage();start=time.monotonic()
        row={'case':case['name'],'task':case['task'],'expected_mechanical':case['mechanical'],'expected_keys':case['keys']}
        unavailable=False
        try:
            contract=await asyncio.wait_for(critic.contract(node),90)
            row.update(contract=contract.model_dump(),**evaluate(case,contract.model_dump()))
        except Exception as exc:
            unavailable=isinstance(exc,ProviderUnavailable)
            row.update(passed=None if unavailable else False,error=type(exc).__name__+': '+str(exc)[:700])
        after=model.usage()
        row.update(seconds=round(time.monotonic()-start,2),usage={k:after[k]-before[k] for k in ['calls','tokens','cost']})
        report['reports'].append(row);save()
        print(json.dumps({k:row.get(k) for k in ['case','passed','seconds','usage','error','issues']},ensure_ascii=False),flush=True)
        if unavailable:
            report['stopped']='Provider unavailable; no further cases attempted';break
    save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live',action='store_true')
    p.add_argument('--model',default='cohere/north-mini-code:free')
    p.add_argument('--provider',default='cohere')
    p.add_argument('--case',action='append',choices=[c['name'] for c in CASES])
    p.add_argument('--output',default='data/eval/contract-eval.json')
    args=p.parse_args()
    if args.live:asyncio.run(run(args))
    else:print('No API calls. Cases:',[c['name'] for c in CASES if not args.case or c['name'] in args.case])
