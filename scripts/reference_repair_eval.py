"""Free-cloud evaluation of the integrated recovery path on seeded failures.

Implementations and incorrect newly added tests are supplied to simulate a paused
development session. This measures repair only, not end-to-end coding ability.
No API calls without --live. Every model route must have zero prices.
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
from textwrap import dedent
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def cases():
    return [dict(name='rolling_total',class_name='RollingTotal',
        task='RollingTotal(width) accepts a positive integer width, excluding bool. add(value) retains only the latest width numeric values. total() returns their sum, or 0 when empty, without changing state. Add regression tests and run checks.',
        source='''class RollingTotal:
    def __init__(self,width):
        if type(width) is not int or width<=0: raise ValueError()
        self.width=width; self.values=[]
    def add(self,value): self.values=(self.values+[value])[-self.width:]
    def total(self): return sum(self.values)
''',original='''    def test_empty(self): self.assertEqual(RollingTotal(2).total(),0)
''',added='''    def test_recent_values(self):
        r=RollingTotal(2)
        r.add(3)
        r.add(6)
        r.add(-2)
        self.assertEqual(r.total(),9)
        r.add(5)
        self.assertIsNone(r.total())
''',acceptance='''import random
from subject import RollingTotal
for bad in [0,-1,True,1.5,None]:
    try: RollingTotal(bad)
    except ValueError: pass
    else: raise AssertionError('capacity validation')
random.seed(23917)
for width in [1,2,5]:
    c=RollingTotal(width); history=[]
    for _ in range(120):
        value=random.randint(-20,20);history.append(value);c.add(value)
        assert c.total()==sum(history[-width:])
        assert c.total()==sum(history[-width:])
print('PASS independent rolling totals')
'''),dict(name='bounded_queue',class_name='BoundedQueue',
        task='BoundedQueue(capacity) accepts a positive integer capacity, excluding bool. push(value) appends a value. If the queue is full, discard the oldest stored value to make room. pop() removes and returns the oldest remaining value, or None when empty. Values may be None. Add regression tests and run checks.',
        source='''class BoundedQueue:
    def __init__(self,capacity):
        if type(capacity) is not int or capacity<=0: raise ValueError()
        self.capacity=capacity; self.values=[]
    def push(self,value):
        if len(self.values)==self.capacity: self.values.pop(0)
        self.values.append(value)
    def pop(self): return self.values.pop(0) if self.values else None
''',original='''    def test_empty(self): self.assertIsNone(BoundedQueue(2).pop())
''',added='''    def test_overflow(self):
        q=BoundedQueue(2)
        q.push('a')
        q.push('b')
        q.push('c')
        self.assertEqual(q.pop(),'a')
        self.assertEqual(q.pop(),'b')
        self.assertEqual(q.pop(),'c')
''',acceptance='''import random
from collections import deque
from subject import BoundedQueue
for bad in [0,-1,True,1.5,None]:
    try: BoundedQueue(bad)
    except ValueError: pass
    else: raise AssertionError('capacity validation')
random.seed(34091)
for capacity in [1,2,5]:
    q=BoundedQueue(capacity);reference=deque(maxlen=capacity)
    for _ in range(120):
        if random.random()<.65:
            value=random.choice([None,0,'a','b']);q.push(value);reference.append(value)
        else: assert q.pop()==(reference.popleft() if reference else None)
print('PASS independent FIFO behavior')
''')]


async def run(args):
    from quality_ladder import source_manifest
    from src.coding.harness import Workspace
    from src.coding.model import OpenRouterModel
    from src.coding.control import SessionControl
    from src.coding.critic import TaskContract
    from src.coding.orchestration import Orchestrator
    from src.coding.supervisor import Supervisor
    from src.coding.test_provenance import TestProvenance
    from src.coding.verifier import isolated_probe
    from src.runtime import hitl
    model=OpenRouterModel({'model':args.model,'provider_only':[args.provider],'max_cost_usd':0,
        'max_input_usd_per_million':0,'max_output_usd_per_million':0,'max_output_tokens':8192,
        'cloud_context_tokens':32768,'min_request_interval_seconds':12})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[1];manifest=source_manifest(project)
    manifest[str(Path(__file__).relative_to(project))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report={'scope':__doc__,'model':model.name,'provider':args.provider,'free_only':True,
        'source_manifest':manifest,'cases':[]}
    hitl.set_work_mode('auto')
    for case in cases():
        if args.case and case['name']!=args.case:continue
        root=output.parent/'quality'/('runtime-'+case['name']+'-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
        directory=root/'.sea';directory.mkdir()
        (root/'subject.py').write_text(case['source'])
        original='import unittest\nfrom subject import '+case['class_name']+'\nclass Tests(unittest.TestCase):\n'+case['original']
        (root/'test_subject.py').write_text(original)
        ws=Workspace(root,directory,{'unit':[sys.executable,'-B','-m','unittest','-q']})
        ws.test_provenance=TestProvenance.start(ws)
        control=SessionControl();control.execution='auto'
        supervisor=Supervisor(directory,control,model,False);control.supervisor=supervisor
        o=Orchestrator(SimpleNamespace(control=control,model=model,config={}),ws,supervisor,
            {'same_model':True,'critic':{'enabled':True,'active':True,'reference_repair':True,'reference_output_tokens':8192,'max_calls':8}})
        node=o.scheduler.nodes['main'];node.task=case['task'];node.memory.plan=['Repair the supplied new regression test and verify']
        o.scheduler.switch('main','repair')
        # The supplied paused state includes a grounded acceptance contract. No
        # claim is made about the cost/quality of generating its preceding work.
        o.critic.contracts[('main',0)]=TaskContract.model_validate({'requirements':[
            {'id':'r1','text':'Add regression tests','validation':'new_or_updated_tests','source_quote':'Add regression tests'},
            {'id':'r2','text':'run checks','validation':'registered_checks','source_quote':'run checks'}]},context={'sources':[case['task']]})
        start=model.usage();started=time.monotonic();row={'case':case['name'],'workspace':str(root)}
        print(json.dumps({'starting':case['name'],'workspace':str(root)}),flush=True)
        try:
            added=await ws.execute('edit',{'path':'test_subject.py','old':original,'new':original+case['added']})
            if not added['ok']:raise ValueError('Could not prepare simulated authored test')
            failed=await ws.execute('check',{'name':'unit'});row['initial_check']=failed
            if failed['ok']:raise ValueError('Case must start with a failing newly added test')
            repair=await o.diagnose_check_failure(node,'unit',ws.record(failed));row['repair']=repair
            if repair and repair.get('stop'):
                row.update(outcome='incomplete_provider_or_budget_failure',error=repair['problem'])
            row['independent_check']=await isolated_probe(ws,case['acceptance'])
            programs=[json.loads(p.read_text()) for p in directory.glob('[0-9]*.json')]
            program=next((p['output'] for p in programs if p.get('kind')=='blind_reference_program'),None)
            if program:
                oracle_root=directory/'independent-reference';oracle_root.mkdir()
                logs=oracle_root/'.sea';logs.mkdir();(oracle_root/'subject.py').write_text(program)
                oracle=Workspace(oracle_root,logs,{})
                try:row['reference_independent_check']=await isolated_probe(oracle,case['acceptance'])
                finally:oracle._scratch.cleanup()
            row['passed']=bool(repair and repair.get('repair_applied') and 'unit' in ws.passed
                and ws.verified_revision==ws.revision and row['independent_check']['ok'])
            row.setdefault('outcome','repair_passed' if row['passed'] else 'repair_not_established')
        except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)[:600]
        finally:
            end=model.usage();row.update(seconds=round(time.monotonic()-started,2),
                usage={k:end[k]-start[k] for k in ['calls','tokens','cost']})
            report['cases'].append(row);report['usage']=end
            report['source_unchanged']=all(hashlib.sha256((project/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            await o.close();ws._scratch.cleanup()
        print(json.dumps({k:row.get(k) for k in ['case','passed','seconds','usage','error']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--model',default='liquid/lfm-2.5-2.6b:free');parser.add_argument('--provider',default='liquid/fp8')
    parser.add_argument('--case',choices=[c['name'] for c in cases()])
    parser.add_argument('--output',default='data/eval/runtime-reference-repair.json');args=parser.parse_args()
    if args.live:asyncio.run(asyncio.wait_for(run(args),180))
    else:print('No API calls. Seeded recovery cases:',[c['name'] for c in cases() if not args.case or c['name']==args.case])
