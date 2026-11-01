"""Same-budget reference repair integrates with the actual agent execution loop."""
import asyncio
import json
import sys

import pytest

from src.coding.harness import CodingHarness
from src.runtime import hitl


@pytest.fixture(autouse=True)
def permissions(monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'manual'})


SOURCE='''class Meter:
    def __init__(self,value): self.balance=value
    def consume(self,n): self.balance-=n; return self.balance
    def read(self): return self.balance
'''
ORIGINAL='''import unittest
from meter import Meter
class Tests(unittest.TestCase):
    def test_original(self):
        self.assertEqual(Meter(7).read(),7)
'''
ADDED='''    def test_added(self):
        meter=Meter(7)
        _=meter.consume(3)
        self.assertEqual(meter.read(),7)
        self.assertIsNone(meter.read())
'''
TASK='Meter stores an initial balance. consume(n) subtracts n and returns the new balance. read() returns the current balance without changing it. Add regression tests and run checks.'


class Model:
    def __init__(self,*,wrong_reference=False,steer=False,unavailable=False):
        self.step=0;self.references=0;self.wrong_reference=wrong_reference;self.steer=steer;self.harness=None
        self.unavailable=unavailable
    async def __call__(self,messages,schema,effort):
        packet=json.loads(messages[-1]['content'])
        if schema.__name__=='TaskContract':
            return schema.model_validate({'requirements':[
                {'id':'r1','text':'Add regression tests','validation':'new_or_updated_tests','source_quote':'Add regression tests'},
                {'id':'r2','text':'run checks','validation':'registered_checks','source_quote':'run checks'}]})
        if schema.__name__=='ReferenceProgram':
            self.references+=1
            if self.unavailable:
                from src.coding.model import ProviderUnavailable
                raise ProviderUnavailable('Provider HTTP 429; resume after availability returns')
            assert set(packet)=={'task','human_steering'} and 'assertEqual' not in messages[-1]['content']
            if self.steer:self.harness.control.post('Stop editing tests.',interrupt=False)
            return schema.model_validate({'code':SOURCE.replace('balance-=n','balance-=n+1') if self.wrong_reference else SOURCE})
        if 'probes' in schema.model_json_schema().get('properties',{}):
            return schema.model_validate({'probes':[],'limitation':'Use registered check evidence; no general proof.'})
        if 'oneOf' not in schema.model_json_schema():
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[
                {'requirement':r,'status':'supported','evidence':packet['valid_evidence_ids'][-1:]}
                for r in ['r1','r2']],'issues':[],'next_mode':'verify'})
        turns=[{'action':'plan','plan':['Add a focused test and verify it']},
            {'action':'tool','tool':'read','args':{'path':'meter.py'}},
            {'action':'tool','tool':'edit','args':{'path':'test_meter.py','old':ORIGINAL,'new':ORIGINAL+ADDED}},
            {'action':'tool','tool':'check','args':{'name':'unit'}},
            {'action':'tool','tool':'check','args':{'name':'unit'}},
            {'action':'finish','summary':'Added the regression test and verified its behavior.'}]
        turn=turns[min(self.step,len(turns)-1)];self.step+=1
        return schema.model_validate(turn)


def run_case(tmp_path,model,*,bad_implementation=False):
    (tmp_path/'meter.py').write_text(SOURCE.replace('balance-=n','balance-=n+2') if bad_implementation else SOURCE)
    (tmp_path/'test_meter.py').write_text(ORIGINAL)
    hitl.set_work_mode('auto')
    harness=CodingHarness(tmp_path,model=model,config={'learn':False,'max_turns':6,
        'checks':{'unit':[sys.executable,'-B','-m','unittest','-q']},
        'orchestration':{'enabled':True,'auto_domains':True,'same_model':True,'observer':False,
            'critic':{'enabled':True,'active':True,'reference_repair':True,'max_calls':8}}})
    model.harness=harness
    result=asyncio.run(harness.run(TASK))
    return harness,result


def test_runtime_repairs_new_expectations_then_finishes_after_real_checks(tmp_path):
    model=Model();harness,result=run_case(tmp_path,model)
    assert result['status']=='done' and result['checks_passed']==['unit']
    text=(tmp_path/'test_meter.py').read_text()
    assert text.startswith(ORIGINAL) and 'self.assertEqual(meter.read(),4)' in text
    assert 'self.assertEqual(meter.read(), 4)' in text and (tmp_path/'meter.py').read_text()==SOURCE
    assert model.references==1
    assert harness.workspace.verified_revision==harness.workspace.revision
    node=harness.control.orchestrator.scheduler.nodes['main']
    assert node.memory.reads['test_meter.py'].find('self.assertEqual(meter.read(),4)')>=0
    failure_file=next((tmp_path/'.sea').rglob('failure-focus-main.json'))
    assert json.loads(failure_file.read_text())=={}


@pytest.mark.parametrize('failure',['bad_reference','bad_implementation','steering'])
def test_disagreement_or_new_instructions_do_not_rewrite_tests(tmp_path,failure):
    model=Model(wrong_reference=failure=='bad_reference',steer=failure=='steering')
    _,result=run_case(tmp_path,model,bad_implementation=failure=='bad_implementation')
    assert result['status']!='done' and model.references==1
    assert (tmp_path/'test_meter.py').read_text()==ORIGINAL+ADDED


def test_provider_failure_checkpoints_without_buying_another_actor_request(tmp_path):
    model=Model(unavailable=True);_,result=run_case(tmp_path,model)
    assert result['status']=='paused' and '429' in result['final_answer']
    assert model.references==1 and model.step==5
    assert (tmp_path/'test_meter.py').read_text()==ORIGINAL+ADDED
