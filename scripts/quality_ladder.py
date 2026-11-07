"""Bounded cloud evaluation across parameterized SWE, math and algorithm families.

No API use without --live. Cases/oracles stay outside the agent workspace. A
reported pass requires independent acceptance, explicit process requirements and
completion, not a model's self-rating. Explanatory prose is not formally proved.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.eval.quality_cases import cases,check_answer
from src.eval.quality_transfer_cases import transfer_cases,check_transfer_answer


def source_manifest(project):
    paths=[*sorted((project/'src').rglob('*.py')),Path(__file__).resolve(),project/'uv.lock']
    return {str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def evaluation_protocol(project):
    paths = ['scripts/quality_ladder.py', 'src/eval/quality_cases.py',
             'src/eval/quality_transfer_cases.py', 'src/coding/verifier.py', 'uv.lock']
    return hashlib.sha256(json.dumps({p: hashlib.sha256((project/p).read_bytes()).hexdigest()
                                      for p in paths}, sort_keys=True).encode()).hexdigest()


def configuration(base,args):
    config=copy.deepcopy(base)
    config.update(backend='openrouter',model=args.model,provider_only=[args.provider],
        max_turns=24,max_total_tokens=80000,max_output_tokens=args.max_output_tokens,
        output_recovery=args.output_recovery,
        structured_output=args.structured_output,provider_retries=args.provider_retries,
        min_request_interval_seconds=args.request_interval,
        max_cost_usd=args.budget_usd,max_input_usd_per_million=.30,
        max_output_usd_per_million=2.,allow_web=False,learn=False,
        action_protocol=args.action_protocol,adaptive_tools=args.adaptive_tools,
        result_memory=args.result_memory,
        native_history=args.native_history,
        context_policy=args.context_policy,
        source_prefetch_bytes=args.source_prefetch_bytes,
        completion_review=args.completion_review,
        planning=args.planning,
        temperature=args.temperature,top_p=args.top_p,
        navigation={'backend':'lexical'},checks={})
    if not args.model.startswith('google/'):
        config.update(max_input_usd_per_million=.17,max_output_usd_per_million=.25)
    if args.free_only:
        # Metadata and routing both must enforce free inference; no paid fallback.
        config.update(max_cost_usd=0,max_input_usd_per_million=0,max_output_usd_per_million=0)
    config['orchestration'].update(enabled=True,auto_domains=True,same_model=True,
        observer=False,local_slots=1,cloud_slots=1,max_nodes=4,max_depth=2,mind_turns=5)
    config['orchestration']['critic'].update(enabled=True,backend='local',active=True,
        max_calls=8,max_probes=2,answer_verification=args.answer_verification)
    return config


async def run(args):
    from src.config.config_paths import load_base_config
    from omegaconf import OmegaConf
    from src.harness import Harness
    from src.coding.verifier import isolated_probe
    from src.runtime import run_context,hitl
    base=OmegaConf.to_container(load_base_config()['coding'],resolve=True)
    selected=(cases(args.seed) if args.suite in {'development','all'} else [])
    selected+=(transfer_cases(args.seed) if args.suite in {'transfer','all'} else [])
    selected=[c for c in selected if not args.case or c.name in args.case]
    if not selected:raise ValueError('No selected cases in this suite')
    answer_check=lambda case,answer: (check_transfer_answer if case.name in
        {'network_reliability','constrained_assignment'} else check_answer)(case,answer)
    code=Path(__file__).resolve().parents[1]/'src'/'coding'
    project=code.parents[1]
    manifest=source_manifest(project)
    source_id=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    report={'schema_version':2,'run_id':uuid.uuid4().hex,
        'evaluation_protocol':evaluation_protocol(project),
        'expected_cases':[c.name for c in selected],
        'scenario':'parameterized_quality_ladder_not_public_benchmark','suite':args.suite,'seed':args.seed,
        'model':args.model,'provider_only':args.provider,'reasoning':args.reasoning,
        'adaptive_tools':args.adaptive_tools,'answer_verification':args.answer_verification,
        'result_memory':args.result_memory,
        'native_history':args.native_history,
        'context_policy':args.context_policy,
        'source_prefetch_bytes':args.source_prefetch_bytes,
        'completion_review':args.completion_review,
        'planning':args.planning,
        'temperature':args.temperature,'top_p':args.top_p,
        'cost_envelope_per_run':0 if args.free_only else args.budget_usd,
        'free_only':args.free_only,'max_output_tokens':args.max_output_tokens,'output_recovery':args.output_recovery,
        'structured_output':args.structured_output,'provider_retries':args.provider_retries,
        'action_protocol':args.action_protocol,
        'request_interval_seconds':args.request_interval,
        'harness_id':hashlib.sha256(b''.join(p.read_bytes() for p in sorted(code.glob('*.py')))).hexdigest(),
        'cases_sha256':hashlib.sha256(json.dumps([vars(c) for c in selected],default=str,sort_keys=True).encode()).hexdigest(),
        'formal_proof_validated':False,'reports':[]}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    snapshot=output.parent/'sources'/(source_id+'.zip')
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():
        with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as archive:
            for name in manifest:archive.write(project/name,name)
    report['core_source_snapshot']=str(snapshot)
    report['source_manifest']=manifest
    report['source_id']=source_id
    report['configuration']=configuration(base,args)
    report['dependency_lock_sha256']=hashlib.sha256((code.parents[1]/'uv.lock').read_bytes()).hexdigest()
    slots=asyncio.Semaphore(args.jobs)
    async def one(case):
        async with slots:
            root=Path(args.work_dir)/(case.name+'-'+str(args.seed)+'-'+uuid.uuid4().hex[:8])
            root.mkdir(parents=True)
            for name,content in case.files.items():(root/name).write_text(content)
            config=configuration(base,args)
            if case.domain=='software':config['checks']={'unit':[sys.executable,'-B','-m','unittest','-q']}
            agent=Harness(root,config=config)
            started=time.monotonic()
            print(json.dumps({'starting':case.name,'workspace':str(root)},ensure_ascii=False),flush=True)
            with run_context.request_scope('quality-'+root.name,'integration'):
                hitl.set_work_mode('auto')
                try:result=await asyncio.wait_for(agent.run(case.task,reasoning=args.reasoning),args.timeout)
                except Exception as exc:
                    o=getattr(agent.control,'orchestrator',None)
                    usage=o.usage() if o else agent.model.usage()
                    result={'status':'error','error':type(exc).__name__+': '+str(exc)[:500],**usage}
                    if hasattr(agent,'token_budget'):
                        result['token_budget']=agent.token_budget.packet()
                o=getattr(agent.control,'orchestrator',None)
                counts=getattr(getattr(agent,'supervisor',None),'tool_counts',{})
                tools={name for names in counts.values() for name in names}
                if case.domain=='software' and o:
                    accepted=await isolated_probe(o.ws,case.acceptance)
                    answer_ok=bool(accepted.get('ok'))
                    process_ok=(any(Path(p).name.startswith('test_') for p in o.ws.changed)
                        and 'unit' in o.ws.passed and o.ws.verified_revision==o.ws.revision)
                else:
                    try:answer_ok=answer_check(case,result.get('final_answer',''))
                    except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError):answer_ok=False
                    process_ok=bool(tools&{'math','experiment','probe','answer'}) and not any(p.name!='.sea' for p in root.iterdir())
                    accepted={'ok':answer_ok,'scope':'Exact finite answer; explanation not formally checked'}
            row={'case':case.name,'domain':case.domain,'workspace':str(root),
                'seconds':round(time.monotonic()-started,2),'answer_passed':answer_ok,
                'process_passed':process_ok,'passed':answer_ok and process_ok and result.get('status')=='done',
                'false_accept':result.get('status')=='done' and not answer_ok,
                'independent_check':accepted,'tools_by_agent':counts,'result':result}
            trajectory = getattr(getattr(agent, 'supervisor', None), 'trajectory', None)
            if trajectory:
                # These are known diagnostic cases, deliberately excluded from training.
                # Agent-authored assertions and model reviews never assign this reward.
                trajectory.evaluation(task_id=case.name+'-'+str(args.seed), family=case.name,
                    split='diagnostic', verifier_id=hashlib.sha256(str(case.acceptance).encode()).hexdigest(),
                    answer_ok=answer_ok, process_ok=process_ok, status=result.get('status'),
                    observation=accepted, eligible=False,
                    source_unchanged=manifest==source_manifest(project),
                    infrastructure_error=not counts and not result.get('tokens', 0))
                result['trajectory'] = trajectory.receipt()
            if case.domain!='software':
                candidates=[]
                for p in root.glob('.sea/coding/*/decisions.jsonl'):
                    for line in p.read_text().splitlines():
                        t=json.loads(line).get('turn',{})
                        if t.get('action')=='finish' and t.get('summary') not in candidates:candidates.append(t.get('summary',''))
                checked=[]
                for candidate in candidates:
                    try:checked.append(answer_check(case,candidate))
                    except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError):checked.append(False)
                row.update(candidate_attempts=len(candidates),candidate_correct_any=any(checked))
            report['reports'].append(row)
            report['reports'].sort(key=lambda r:r['case'])
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(json.dumps({k:row[k] for k in ('case','seconds','passed','answer_passed','process_passed')}
                |{k:result.get(k) for k in ('status','cost','tokens','error')}),flush=True)
    await asyncio.gather(*(one(c) for c in selected))
    report['source_unchanged_during_run']=manifest==source_manifest(project)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--model',default='google/gemini-3.1-flash-lite')
    parser.add_argument('--provider',default='google-ai-studio')
    parser.add_argument('--reasoning',choices=['low','medium','high'],default='low')
    parser.add_argument('--adaptive-tools',action='store_true')
    parser.add_argument('--result-memory',action='store_true',help='Retain bounded exact tool observations beyond the rolling action window')
    parser.add_argument('--native-history',action='store_true',help='Retain bounded actual native exchanges and provider-returned reasoning per actor')
    parser.add_argument('--context-policy',choices=['bounded','single_action'],default='bounded',
        help='single_action retains the latest observation and a continuation brief; older state is retrieved from disk')
    parser.add_argument('--source-prefetch-bytes',type=int,choices=range(4801),default=0,
        metavar='0..4800',help='Read a bounded initial source context without model calls; 0 disables this experiment')
    parser.add_argument('--completion-review',action='store_true',
        help='Propose delivery after current checks pass; normal independent completion review is still required')
    parser.add_argument('--planning',choices=['required','adaptive'],default='required',help='Whether auto execution needs a separate model plan action; explicit plan-only mode stays read-only')
    parser.add_argument('--temperature',type=float,help='Explicit sampling temperature; omitted uses the provider default')
    parser.add_argument('--top-p',type=float,help='Explicit nucleus sampling; omitted uses the provider default')
    parser.add_argument('--answer-verification',action='store_true')
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--suite',choices=['development','transfer','all'],default='development')
    parser.add_argument('--case',action='append',choices=[c.name for c in cases()+transfer_cases()])
    parser.add_argument('--budget-usd',type=float,default=.05)
    parser.add_argument('--free-only',action='store_true',help='Require zero input/output/request prices; no paid routing')
    parser.add_argument('--max-output-tokens',type=int,default=3500)
    parser.add_argument('--output-recovery',action='store_true',help='Allow one bounded action-completion call using returned draft text')
    parser.add_argument('--structured-output',choices=['auto','json_schema','json_object','text'],default='auto')
    parser.add_argument('--action-protocol',choices=['json','tools'],default='json')
    parser.add_argument('--provider-retries',type=int,choices=range(9),default=4,help='Shared per-engine allowance for transient HTTP failures; 0 disables retries')
    parser.add_argument('--request-interval',type=float,default=0,help='Minimum seconds between request starts; does not retry failed requests')
    parser.add_argument('--jobs',type=int,choices=[1,2],default=1)
    parser.add_argument('--timeout',type=int,default=150)
    parser.add_argument('--work-dir',default='data/eval/quality')
    parser.add_argument('--output',default='data/eval/quality-ladder.json')
    return parser


if __name__=='__main__':
    parser=build_parser()
    args=parser.parse_args()
    if not 0<args.budget_usd<=.10:parser.error('budget must be in (0,.10] per run')
    if not 10<=args.timeout<=300:parser.error('timeout must be in [10,300] seconds')
    if not 256<=args.max_output_tokens<=8192:parser.error('max-output-tokens must be in [256,8192]')
    if not 0<=args.request_interval<=30:parser.error('request-interval must be in [0,30] seconds')
    if args.live:raise SystemExit(asyncio.run(run(args)))
    selected=(cases(args.seed) if args.suite in {'development','all'} else [])
    selected+=(transfer_cases(args.seed) if args.suite in {'transfer','all'} else [])
    print('No API calls. Selected cases:',[c.name for c in selected if not args.case or c.name in args.case])
