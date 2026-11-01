"""Opt-in paired cloud-9B integration: passive review, active probes, typed routing.

Three small synthetic cases are an engineering check, not a public benchmark or a
statistically reliable quality estimate. Acceptance code is withheld until each run ends.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

VARIANTS=('passive','active','hybrid')
FILES={
    'ports.py':'def parse_port(text):\n    return int(text)\n',
    'test_ports.py':'''import unittest
from ports import parse_port
class Ports(unittest.TestCase):
    def test_existing(self):
        self.assertEqual(parse_port('8080'),8080)
''',
}
CASES=[
    ('software','Harden ports.parse_port(text), preserving its public signature. Accept only nonempty strings '
     'consisting of ASCII digits, with numeric value 1..65535 inclusive; leading zeros are allowed. '
     'Reject every other input (including non-strings) with ValueError. First delegate one independent '
     'read-only review of the input contract and boundary cases. Then implement the fix in the parent, '
     'add focused regression tests and run the registered unit check. Keep changes small.'),
    ('mathematics','Compute exactly sum(k*(k+1)/((k+2)*(k+3)) for k=1..35). '
     'Use an exact computation tool and explain a derivation. Do not create project files. '
     'Return a JSON final answer with keys value (a reduced fraction p/q) and derivation (string).'),
    ('algorithms','For unlimited coins [1,5,8], find the smallest positive amount where largest-first '
     'greedy uses more coins than an optimal solution. Use an executable finite experiment. Explain '
     'a general optimal algorithm, its invariant and complexity. Do not create project files. '
     'Return a JSON final answer with keys amount (integer), greedy (list of coin values), '
     'optimal (list of coin values), algorithm (string), correctness (string), complexity (string).'),
]
ACCEPTANCE='''from ports import parse_port
for text in ('1','65535','00080','8080'):
    assert parse_port(text)==int(text),repr(text)
for text in ('','0','65536','-1','+80',' 80','80 ','1_0','１２','١٢','1.5',None,80,True,[],{}):
    try: parse_port(text)
    except ValueError: pass
    else: raise AssertionError(repr(text))
print('PASS: valid boundaries, zeros, ASCII-only, types and invalid syntax')
'''

TRANSFER_FILES={
    'batches.py':'def chunks(items,size):\n    return [items[:size]]\n',
    'test_batches.py':'import unittest\nfrom batches import chunks\nclass Batches(unittest.TestCase):\n    def test_one(self):\n        self.assertEqual(chunks([1,2],2),[[1,2]])\n',
}
TRANSFER_CASES=[
    ('software','Fix batches.chunks(items,size). Given a list and a positive integer size, return consecutive '
     'nonempty chunks containing every input item in order, with at most size items in each chunk. '
     'Empty input returns []. Do not mutate the input. Reject zero, negative, boolean and non-integer sizes '
     'with ValueError. First delegate one independent read-only review of boundary cases. Implement the fix '
     'in the parent, add focused regression tests and run the registered unit check.'),
    ('mathematics','Compute exactly sum(1/(k*(k+1)*(k+2)) for k=1..48). Use an exact computation tool and '
     'give a derivation. Do not create project files. Return JSON with keys value (a reduced fraction p/q) '
     'and derivation (string).'),
    ('algorithms','For intervals [(0,4),(3,5),(4,7),(5,9),(7,10)], find a largest subset of non-overlapping '
     'intervals; touching endpoints are allowed. Use an executable exhaustive experiment to check the result. '
     'Explain a general optimal algorithm, its correctness argument and complexity. Do not create project files. '
     'Return JSON with keys selected (list of [start,end] pairs), count (integer), algorithm (string), '
     'correctness (string), complexity (string).'),
]
TRANSFER_ACCEPTANCE='''from batches import chunks
for n in range(10):
    for size in range(1,7):
        items=list(range(n));before=list(items);result=chunks(items,size)
        assert result==[items[i:i+size] for i in range(0,n,size)],(n,size,result)
        assert items==before
for size in (0,-1,True,False,1.5,'2',None):
    try: chunks([1,2],size)
    except ValueError: pass
    else: raise AssertionError(repr(size))
print('PASS: complete ordered partition, boundaries, input preservation, size validation')
'''


def answer_check(case,answer,suite='diagnostic'):
    value=json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
    if case=='mathematics':
        expected=(sum((Fraction(1,k*(k+1)*(k+2)) for k in range(1,49)),Fraction()) if suite=='transfer' else
                  sum((Fraction(k*(k+1),(k+2)*(k+3)) for k in range(1,36)),Fraction()))
        return Fraction(value['value'])==expected and bool(value['derivation'])
    if suite=='transfer':
        import itertools
        universe={(0,4),(3,5),(4,7),(5,9),(7,10)}
        selected=sorted(tuple(x) for x in value['selected'])
        compatible=lambda xs:all(a[1]<=b[0] for a,b in zip(sorted(xs),sorted(xs)[1:]))
        optimum=max(len(xs) for n in range(len(universe)+1) for xs in itertools.combinations(universe,n) if compatible(xs))
        return (len(set(selected))==len(selected)==value['count']==optimum and set(selected)<=universe
                and compatible(selected) and all(value.get(k) for k in ('algorithm','correctness','complexity')))
    return (value['amount']==10 and value['greedy']==[8,1,1]
            and sorted(value['optimal'])==[5,5]
            and all(value.get(k) for k in ('algorithm','correctness','complexity')))


def configuration(base,variant,model,budget=.025):
    config=copy.deepcopy(base)
    config.update(backend='openrouter',model=model,max_turns=16,max_output_tokens=2500,
                  max_cost_usd=budget-.005 if variant=='hybrid' else budget,max_total_tokens=60000,
                  max_input_usd_per_million=.17,max_output_usd_per_million=.25,
                  allow_web=False,learn=False,checks={})
    if model.startswith('google/'):
        config.update(max_input_usd_per_million=.30,max_output_usd_per_million=2.0)
    config['navigation']={'backend':'lexical'}
    config['orchestration'].update(enabled=True,auto_domains=True,same_model=variant!='hybrid',
                                  observer=False,local_slots=1,max_nodes=4,max_depth=2,mind_turns=5)
    config['orchestration']['critic'].update(enabled=True,backend='local',active=variant!='passive',
                                           max_calls=8,max_probes=2)
    config['orchestration']['decision'].update(backend='decisions',max_calls=4,max_cost_usd=.001,
        planner={'model':model,'max_calls':2,'max_cost_usd':.004,
                 'max_input_usd_per_million':config['max_input_usd_per_million'],
                 'max_output_usd_per_million':config['max_output_usd_per_million']})
    return config


async def run(args):
    from src.harness import Harness
    from src.coding.verifier import isolated_probe
    from src.config.config_paths import load_base_config
    from src.runtime import hitl,run_context
    from omegaconf import OmegaConf
    base=OmegaConf.to_container(load_base_config()['coding'],resolve=True)
    cases=TRANSFER_CASES if args.suite=='transfer' else CASES
    files=TRANSFER_FILES if args.suite=='transfer' else FILES
    acceptance=TRANSFER_ACCEPTANCE if args.suite=='transfer' else ACCEPTANCE
    code=Path(__file__).resolve().parents[1]/'src'/'coding'
    report={'scenario':'three_domain_ablation_not_benchmark','model':args.model,'reasoning':args.reasoning,
            'provider_only':args.provider,
            'action_protocol':args.protocol,
            'suite':args.suite,
            'harness_id':hashlib.sha256(b''.join(p.read_bytes() for p in sorted(code.glob('*.py')))).hexdigest(),
            'cases_sha256':hashlib.sha256(json.dumps(cases).encode()).hexdigest(),
            'cost_envelope_per_run':args.budget_usd,'proof_validated':False,'reports':[]}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    for index,(name,task) in enumerate(cases):
        if args.case and name not in args.case: continue
        variants=VARIANTS[index:]+VARIANTS[:index]
        for variant in variants:
            if args.variant and variant not in args.variant: continue
            root=Path(args.work_dir)/(name+'-'+variant+'-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
            if name=='software':
                for filename,content in files.items(): (root/filename).write_text(content)
            config=configuration(base,variant,args.model,args.budget_usd)
            config['action_protocol']=args.protocol
            if args.provider:
                config['provider_only']=[args.provider]
                config['orchestration']['decision']['planner']['provider_only']=[args.provider]
            if name=='software': config['checks']={'unit':[sys.executable,'-B','-m','unittest','-q']}
            agent=Harness(root,config=config)
            started=time.monotonic()
            print(json.dumps({'starting':name,'variant':variant,'workspace':str(root)}),flush=True)
            with run_context.request_scope('ablation-'+name+'-'+variant,'integration'):
                hitl.set_work_mode('auto')
                try:
                    result=await asyncio.wait_for(agent.run(task,reasoning=args.reasoning),120)
                except Exception as exc:
                    orchestrator=getattr(agent.control,'orchestrator',None)
                    usage=orchestrator.usage() if orchestrator else agent.model.usage()
                    result={'status':'error','error':type(exc).__name__+': '+str(exc)[:500],**usage}
                orchestrator=getattr(agent.control,'orchestrator',None)
                counts=getattr(getattr(agent,'supervisor',None),'tool_counts',{})
                all_tools={tool for tools in counts.values() for tool in tools}
                if name=='software' and orchestrator:
                    checked=await isolated_probe(orchestrator.ws,acceptance)
                    passed=bool(checked.get('ok'))
                    process_ok=(any(n.id!='main' for n in orchestrator.scheduler.nodes.values())
                                and any(n.startswith('test') for n in orchestrator.ws.changed)
                                and 'unit' in orchestrator.ws.passed)
                else:
                    try: passed=answer_check(name,result.get('final_answer',''),args.suite)
                    except (ValueError,TypeError,KeyError,ZeroDivisionError): passed=False
                    checked={'ok':passed,'scope':'Exact numerical answer; written derivation is not formally verified'}
                    process_ok=(bool(all_tools & ({'math','experiment','probe'} if name=='mathematics' else {'experiment','probe'}))
                                and not any(p for p in root.iterdir() if p.name!='.sea'))
            row={'case':name,'variant':variant,'workspace':str(root),'seconds':round(time.monotonic()-started,2),
                 'answer_passed':passed,'process_passed':process_ok,'independent_check':checked,
                 'passed':passed and process_ok and result.get('status')=='done',
                 'controller_usage':orchestrator.router.usage() if orchestrator else {},'result':result}
            row['tools_by_agent']=counts
            # Generator success and selection/completion success are different measurements.
            # This does not give the model access to the independent answer checker.
            candidates=[]
            trace=getattr(getattr(agent,'workspace',None),'directory',root)/'decisions.jsonl'
            if name!='software' and trace.exists():
                for line in trace.read_text().splitlines():
                    turn=json.loads(line).get('turn',{})
                    if turn.get('action')!='finish' or turn.get('summary') in candidates: continue
                    candidates.append(turn.get('summary',''))
                checks=[]
                for candidate in candidates:
                    try: checks.append(answer_check(name,candidate,args.suite))
                    except (ValueError,TypeError,KeyError,ZeroDivisionError): checks.append(False)
                row['candidate_attempts']=len(candidates)
                row['candidate_correct_any']=any(checks)
                row['false_accept']=result.get('status')=='done' and not passed
            report['reports'].append(row)
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(json.dumps({k:row[k] for k in ('case','variant','seconds','answer_passed','process_passed','passed')}
                             |{k:result.get(k) for k in ('status','cost','tokens','error')},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--model',default='qwen/qwen3.5-9b')
    parser.add_argument('--suite',choices=['diagnostic','transfer'],default='diagnostic')
    parser.add_argument('--reasoning',choices=['low','medium','high'],default='low')
    parser.add_argument('--provider',help='Optional exact OpenRouter provider slug; disables provider fallbacks')
    parser.add_argument('--protocol',choices=['json','tools'],default='json')
    parser.add_argument('--budget-usd',type=float,default=.025,help='Total envelope per run, including controller/planner')
    parser.add_argument('--case',action='append',choices=[c[0] for c in CASES])
    parser.add_argument('--variant',action='append',choices=VARIANTS)
    parser.add_argument('--work-dir',default='data/eval/ablation')
    parser.add_argument('--output',default='data/eval/harness-ablation.json')
    args=parser.parse_args()
    if not .005<args.budget_usd<=.10: parser.error('--budget-usd must be greater than .005 and at most .10')
    if args.live: raise SystemExit(asyncio.run(run(args)))
    print(f'No API calls. Use --live for bounded cloud integration runs (maximum ${9*args.budget_usd:.3f} for all nine).')
