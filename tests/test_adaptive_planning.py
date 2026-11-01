"""Optional model planning never relaxes operator modes or completion evidence."""
import asyncio
import json
import sys

import pytest

from src.coding.harness import CodingHarness, Turn, Workspace
from src.coding.control import SessionControl
from src.coding.supervisor import Supervisor
from src.coding.orchestration import Orchestrator
from src.coding.decision import Branch
from src.coding.request import compact_packet
from src.runtime import hitl


class Scripted:
    def __init__(self,*turns):
        self.turns=list(turns)
        self.requests=[]

    async def __call__(self,messages,schema,effort):
        self.requests.append((messages,schema.model_json_schema()))
        return schema.model_validate(self.turns.pop(0))


def action(tool_name,**args):
    return {'action':'tool','tool':tool_name,'args':args}


def config(**extra):
    return {'planning':'adaptive','learn':False,'max_turns':8,
            'orchestration':{'enabled':True,'same_model':True,'critic':{'enabled':False}},**extra}


@pytest.fixture(autouse=True)
def permissions(monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'auto'})


def test_auto_edits_and_checks_without_fabricating_a_plan(tmp_path):
    (tmp_path/'value.py').write_text('VALUE=1\n')
    model=Scripted(action('read',path='value.py'),
        action('edit',path='value.py',old='VALUE=1',new='VALUE=2'),
        {'action':'finish','summary':'premature'},action('check',name='unit'),
        {'action':'finish','summary':'Updated and checked'})
    h=CodingHarness(tmp_path,model=model,config=config(checks={
        'unit':[sys.executable,'-B','-c','from value import VALUE; assert VALUE==2']}))
    result=asyncio.run(h.run('Set VALUE=2 and verify'))
    assert result['status']=='done' and result['plan']==[]
    assert result['checks_passed']==['unit'] and (tmp_path/'value.py').read_text()=='VALUE=2\n'
    assert any('Completion rejected' in str(m) for m,_ in model.requests)
    initial=model.requests[0][1]
    assert any('edit' in b['properties'].get('tool',{}).get('enum',[]) for b in initial['oneOf'])


def test_explicit_plan_mode_remains_nonexecuting_with_adaptive_planning(tmp_path):
    class Malicious(Scripted):
        async def __call__(self,messages,schema,effort):
            # Exercise runtime guards even if a model adapter bypasses decoding.
            return Turn.model_validate(self.turns.pop(0))
    model=Malicious(action('write',path='forbidden.py',content='bad'),
        action('run',argv=[sys.executable,'-c','raise Exception']),
        {'action':'plan','plan':['Inspect only; do not execute']})
    result=asyncio.run(CodingHarness(tmp_path,model=model,config=config()).run('Plan changes',execution='plan'))
    assert result['status']=='planned' and not result['tools_used']
    assert not (tmp_path/'forbidden.py').exists()


def test_adaptive_does_not_finish_after_unverified_change(tmp_path):
    model=Scripted(action('write',path='value.py',content='VALUE=2'),
                   {'action':'finish','summary':'unverified'})
    result=asyncio.run(CodingHarness(tmp_path,model=model,config=config(max_turns=2,checks={})).run('Create a module'))
    assert result['status']=='paused' and not result['checks_passed']


def test_adaptive_worker_is_isolated_and_still_needs_checks(tmp_path):
    (tmp_path/'value.py').write_text('VALUE=1\n')
    directory=tmp_path/'.sea'/'coding'/'test';directory.mkdir(parents=True)
    checks={'unit':[sys.executable,'-B','-c','from value import VALUE; assert VALUE==2']}
    ws=Workspace(tmp_path,directory,checks)
    control=SessionControl();control.execution='auto'
    model=Scripted(action('edit',path='value.py',old='VALUE=1',new='VALUE=2'),
        {'action':'finish','summary':'premature'},action('check',name='unit'),
        {'action':'finish','summary':'Child updated and checked'})
    supervisor=Supervisor(directory,control,model,False)
    h=CodingHarness(tmp_path,model=model,config=config());h.control=control
    o=Orchestrator(h,ws,supervisor,{'same_model':True,'isolated_writes':True,'critic':{'enabled':False}})
    node=o.scheduler.create('main',Branch(mode='implement',task='Set VALUE=2',tools=['read','edit','check']))
    result=asyncio.run(o.run_mind(node))
    assert result['proposal']['files']==['value.py'] and not node.memory.plan
    assert 'VALUE=1' in (tmp_path/'value.py').read_text()  # No implicit parent adoption.
    assert any('Run registered checks' in str(m) for m,_ in model.requests)
    o.branches.close()


def test_invalid_planning_setting_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='planning must'):
        CodingHarness(tmp_path,model=Scripted(),config={'planning':'unbounded'})


def test_quote_compaction_keeps_information_outside_transmitted_task_and_steering():
    task='Preserve the API. Add focused tests.'
    packet={'task':task,'human_steering':[{'id':'u1','text':'Do not access the network.'}],
            'obligations':{'requirements':[
                {'id':'r1','text':'Preserve the API','source_quote':'Preserve the API.'},
                {'id':'r2','text':'Keep networking off','source_quote':'Do not access the network.'},
                {'id':'r3','text':'Preserve a detail beyond truncated task','source_quote':'Keep the original Unicode normalization semantics.'}],
                'ready':False},'verified':False}
    compact=compact_packet(packet)
    rules=compact['obligations']['requirements']
    assert 'source_quote' not in rules[0] and 'source_quote' not in rules[1]
    assert rules[2]['source_quote']==packet['obligations']['requirements'][2]['source_quote']
    assert compact['task']==task and compact['human_steering']==packet['human_steering']
    assert compact['verified'] is False and compact['obligations']['ready'] is False
    assert packet['obligations']['requirements'][0]['source_quote']=='Preserve the API.'
