"""Diagnostics observe exact test operations, without promoting assertions to truth."""
import asyncio
import json
import sys

import pytest

from src.coding.control import SessionControl
from src.coding.critic import ContextCritic
from src.coding.failure_diagnosis import diagnose_failure,diagnostic_packet
from src.coding.harness import Workspace
from src.coding.minds import MindScheduler
from src.coding.supervisor import Supervisor
from src.coding.test_trace import trace_failed_test,target_test
from src.runtime import hitl


@pytest.fixture(autouse=True)
def permissions(monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'manual'})


SOURCE='''class Meter:
    def __init__(self): self.balance=0
    def deposit(self,amount): self.balance+=amount
    def consume(self,amount):
        self.balance-=amount
        return self.balance
'''
TEST='''import unittest
from meter import Meter
class TestMeter(unittest.TestCase):
    def test_consumption(self):
        meter=Meter()
        meter.deposit(7)
        self.assertEqual(meter.consume(3),7)
'''


def workspace(tmp_path):
    directory=tmp_path/'.sea';directory.mkdir()
    (tmp_path/'meter.py').write_text(SOURCE)
    (tmp_path/'test_meter.py').write_text(TEST)
    ws=Workspace(tmp_path,directory,{'unit':[sys.executable,'-B','-m','unittest','-q']})
    ws.changed.add('test_meter.py')
    hitl.set_work_mode('auto')
    return ws


def test_exact_failed_test_trace_keeps_the_mutation_and_actual_result(tmp_path):
    async def run():
        ws=workspace(tmp_path)
        failure=await ws.execute('check',{'name':'unit'})
        assert not failure['ok']
        value=await trace_failed_test(ws,'unit',failure)
        assert value['ok']
        trace=value['observation']
        assert trace['tests_run']==1 and trace['observed_test_failed'] and trace['omitted_events']==0
        consume=next(e for e in trace['events'] if e.get('call')=='consume')
        assert consume['in']=={'amount':3} and consume['out']==4
        assert consume['state_after']['balance']==4
        assert trace['failures'][0]['message']=='AssertionError: 4 != 7'
        assert not ws.passed and ws.verified_revision==-1
        assert (tmp_path/'meter.py').read_text()==SOURCE
    asyncio.run(run())


def test_diagnosis_is_generated_after_exact_trace_with_no_extra_model_call(tmp_path):
    class Model:
        calls=0
        async def __call__(self,messages,schema,effort):
            self.calls+=1
            packet=json.loads(messages[-1]['content'])
            observed=packet['test_trace']
            assert observed['ok'] and observed['evidence_id'] in packet['valid_evidence_ids']
            assert observed['observation']['observed_test_failed']
            assert not packet['isolated_python_probe_available']
            assert schema.model_json_schema()['properties']['probe_code']['const']==''
            return schema.model_validate({'category':'test_expectation',
                'reason':'Consumption subtracts three from seven; the new test expects the old balance.',
                'evidence':[observed['evidence_id']],'next_steps':['Check the required subtraction before correcting the new assertion.'],
                'probe_code':''})
    async def run():
        ws=workspace(tmp_path);model=Model();control=SessionControl()
        supervisor=Supervisor(ws.directory,control,model,False)
        node=MindScheduler(control,supervisor,{},ws).nodes['main']
        node.task='Consumption subtracts the amount from the balance; add tests.'
        node.memory.files={'meter.py':ws.file_digest('meter.py')}
        critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True})
        failure=await ws.execute('check',{'name':'unit'})
        report=await diagnose_failure(critic,node,ws,'unit',ws.record(failure))
        assert model.calls==critic.calls==1
        assert report['diagnostic_observations']==[] and report['test_trace']['ok']
        assert diagnostic_packet(report)['category']=='test_expectation'
        assert not ws.passed and report['status']=='unverified_hypothesis'
    asyncio.run(run())


@pytest.mark.parametrize('argv,output',[
    (['python','-m','pytest'],'FAIL: test_a (test_a.Test.test_a)'),
    (['python','-m','unittest'],'FAIL: test_a (test_a.Test.test_b)'),
    (['python','-m','unittest'],'FAIL: test_a (../outside.Test.test_a)'),
    (['python','-m','unittest'],'FAILED tests/test_a.py::test_a'),
])
def test_unsupported_or_ambiguous_targets_are_not_transcribed(argv,output):
    assert target_test(argv,output) is None


def test_unavailable_test_module_is_not_imported(tmp_path):
    ws=workspace(tmp_path)
    assert asyncio.run(trace_failed_test(ws,'unit',{'output':'FAIL: test_a (outside.Test.test_a)'})) is None


def test_trace_does_not_evaluate_user_repr_or_dict_property(tmp_path):
    async def run():
        ws=workspace(tmp_path)
        source='''class Bomb:
    @property
    def __dict__(self):
        raise RuntimeError('must not inspect this property')
    def __repr__(self):
        raise RuntimeError('must not call repr')
def identity(value): return value
'''
        (tmp_path/'meter.py').write_text(source)
        (tmp_path/'test_meter.py').write_text('''import unittest
from meter import Bomb,identity
class TestMeter(unittest.TestCase):
    def test_consumption(self):
        value=Bomb()
        identity(value)
        self.assertEqual(4,7)
''')
        failure=await ws.execute('check',{'name':'unit'})
        value=await trace_failed_test(ws,'unit',failure)
        assert value['ok'] and value['observation']['omitted_events']==0
        identity=next(e for e in value['observation']['events'] if e.get('call')=='identity')
        assert identity['in']=={'value':'<Bomb>'} and identity['out']=='<Bomb>'
        assert (tmp_path/'meter.py').read_text()==source
    asyncio.run(run())


def test_trace_redacts_common_credential_fields_without_hiding_ordinary_keys(tmp_path):
    async def run():
        ws=workspace(tmp_path)
        (tmp_path/'meter.py').write_text('''class Meter:
    def __init__(self):
        self.api_key='test-credential-not-for-context'
        self.balance=4
    def read(self,password): return self.balance
''')
        (tmp_path/'test_meter.py').write_text('''import unittest
from meter import Meter
class TestMeter(unittest.TestCase):
    def test_consumption(self):
        meter=Meter()
        self.assertEqual(meter.read('test-password-not-for-context'),7)
''')
        failure=await ws.execute('check',{'name':'unit'})
        value=await trace_failed_test(ws,'unit',failure)
        assert value['ok']
        encoded=json.dumps(value['observation'])
        assert 'test-credential-not-for-context' not in encoded and 'test-password-not-for-context' not in encoded
        read=next(e for e in value['observation']['events'] if e.get('call')=='read')
        assert read['state_after']=={'api_key':'<redacted>','balance':4}
        assert read['in']=={'password':'<redacted>'}
    asyncio.run(run())
