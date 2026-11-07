"""Post-check delivery proposals never replace actual completion gates."""
import asyncio
import json
import sys

import pytest

from src.coding.harness import CodingHarness
from src.runtime import hitl


@pytest.fixture(autouse=True)
def permissions(monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'auto'})


class Model:
    def __init__(self,root,*,critic_ready=True,proposal='ready',json_answer=False):
        self.root=root;self.critic_ready=critic_ready;self.proposal=proposal;self.json_answer=json_answer
        self.actions=0;self.proposals=0;self.reviews=0
        self.actor_packets=[]

    async def __call__(self,messages,schema,effort):
        packet=json.loads(messages[-1]['content'])
        if schema.__name__=='TaskContract':
            return schema.model_validate({'domains':['software'],
                'response_format':'json' if self.json_answer else 'text',
                'response_keys':['value'] if self.json_answer else [],'requirements':[
                    {'id':'r1','text':'Set VALUE=2','validation':'model_evidence'},
                    {'id':'r2','text':'run checks','validation':'registered_checks','source_quote':'run checks'}]})
        if schema.__name__=='DeliveryProposal':
            self.proposals+=1
            assert packet['verified'] and not packet.get('completion_blockers')
            if self.proposal=='defer':
                return schema.model_validate({'ready':False,'summary':None,'remaining_work':'Inspect the API before proposing completion.'})
            if self.proposal=='race':(self.root/'value.py').write_text('VALUE=9\n')
            if self.json_answer:
                shape=schema.model_json_schema()['properties']['summary']['anyOf'][0]
                assert shape['type']=='object' and shape['required']==['value']
            return schema.model_validate({'ready':True,
                'summary':{'value':2} if self.json_answer else 'Set VALUE=2; unit passed.', 'remaining_work':''})
        if 'oneOf' not in schema.model_json_schema():
            self.reviews+=1
            return schema.model_validate({'aligned':True,'ready':self.critic_ready,
                'criteria':[{'requirement':rid,'status':'supported' if self.critic_ready else 'missing',
                             'evidence':packet['valid_evidence_ids'][-1:]} for rid in ('r1','r2')],
                'issues':[] if self.critic_ready else ['Required behavior still missing.'],'next_mode':'implement'})
        actions=[{'action':'tool','tool':'edit','args':{'path':'value.py','old':'VALUE=1','new':'VALUE=2'}},
                 {'action':'tool','tool':'check','args':{'name':'unit'}}]
        turn=actions[self.actions] if self.actions<2 else {'action':'tool','tool':'read','args':{'path':'value.py'}}
        self.actor_packets.append(packet)
        self.actions+=1;return schema.model_validate(turn)


def run(root,model,turns=3):
    (root/'value.py').write_text('VALUE=1\n')
    h=CodingHarness(root,model=model,config={'completion_review':True,'planning':'adaptive',
        'max_turns':turns,'learn':False,'checks':{'unit':[sys.executable,'-B','-c',
            'from value import VALUE; assert VALUE==2']},
        'orchestration':{'enabled':True,'auto_domains':True,'same_model':True,
                         'critic':{'enabled':True,'active':False,'answer_verification':True}}})
    task='Set VALUE=2 and run checks.'+(' Return JSON with integer value.' if model.json_answer else '')
    return asyncio.run(h.run(task)),h


@pytest.mark.parametrize('json_answer',[False,True])
def test_verified_proposal_uses_normal_critic_and_answer_contract(tmp_path,json_answer):
    model=Model(tmp_path,json_answer=json_answer)
    result,h=run(tmp_path,model)
    assert result['status']=='done' and model.proposals==model.reviews==1
    assert result['checks_passed']==['unit'] and result['tools_used']==['edit','check']
    if json_answer:assert json.loads(result['final_answer'])=={'value':2}
    assert h.control.orchestrator.last_assessment['ready']


def test_passing_checks_and_ready_proposal_cannot_override_critic_rejection(tmp_path):
    model=Model(tmp_path,critic_ready=False)
    result,h=run(tmp_path,model,4)
    assert result['status']=='paused' and model.proposals==model.reviews==1
    assert not h.control.orchestrator.last_assessment['ready']


def test_deferred_delivery_returns_to_work_without_repeating_same_proposal(tmp_path):
    model=Model(tmp_path,proposal='defer')
    result,h=run(tmp_path,model,5)
    assert result['status']=='paused' and model.proposals==1 and model.reviews==0
    assert result['tools_used']==['edit','check','read','read']
    assert model.actor_packets[2]['phase']=='implement'
    assert model.actor_packets[2]['cognitive_mode']=='implement'
    assert 'Inspect the API' in str(model.actor_packets[2]['recent'])


def test_source_race_discards_proposal_and_invalidates_old_checks(tmp_path):
    model=Model(tmp_path,proposal='race')
    result,h=run(tmp_path,model)
    assert result['status']=='paused' and model.proposals==1 and model.reviews==0
    assert result['checks_passed']==[] and h.workspace.verified_revision==-1


def test_multifile_implementation_keeps_write_tools_and_requires_final_check(tmp_path):
    class MultiFile:
        calls=0
        async def __call__(self,messages,schema,effort):
            packet=json.loads(messages[-1]['content'])
            turns=[{'action':'tool','tool':'write','args':{'path':'one.py','content':'VALUE=1\n'}},
                   {'action':'tool','tool':'write','args':{'path':'two.py','content':'VALUE=2\n'}},
                   {'action':'finish','summary':'Premature; two files exist'},
                   {'action':'tool','tool':'check','args':{'name':'unit'}},
                   {'action':'finish','summary':'Both files created and checked.'}]
            if self.calls==1:
                assert packet['phase']==packet['cognitive_mode']=='implement'
            if self.calls==3:
                assert not packet['verified']
                assert packet['phase']=='verify'
            result=schema.model_validate(turns[self.calls]);self.calls+=1;return result
    h=CodingHarness(tmp_path,model=MultiFile(),config={'planning':'adaptive','max_turns':5,'learn':False,
        'checks':{'unit':[sys.executable,'-B','-c','import one,two; assert one.VALUE+two.VALUE==3']},
        'orchestration':{'enabled':True,'same_model':True,'critic':{'enabled':False}}})
    result=asyncio.run(h.run('Create one.py and two.py, then verify the pair.'))
    assert result['status']=='done' and result['tools_used']==['write','write','check']
    assert result['checks_passed']==['unit']
