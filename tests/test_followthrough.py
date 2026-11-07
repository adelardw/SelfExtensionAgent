"""Completion repair, grounded grouped outputs, and durable operator handoffs."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.coding.attention import Attention
from src.coding.artifacts import inspect_artifact
from src.coding.control import SessionControl
from src.coding.critic import Assessment, TaskContract, ContextCritic
from src.coding.followthrough import agenda
from src.coding.harness import CodingHarness, Workspace
from src.coding.supervisor import Supervisor
from src.interface import clarify
from src.runtime import hitl


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'auto'})
    monkeypatch.setattr(clarify,'_clarifier',None)
    clarify.reset_ledger()


def workspace(root):
    directory=root/'.sea';directory.mkdir(exist_ok=True)
    return Workspace(root,directory,{})


def test_grouped_artifacts_cannot_validate_only_generator():
    task='Create build.py, report.pdf and slides.pptx.'
    raw={'requirements':[{'id':'r1','text':task,'validation':'artifact_exists',
                         'path':'build.py','source_quote':task}]}
    with pytest.raises(ValidationError,match='omits named files'):
        TaskContract.model_validate(raw,context={'sources':[task]})
    raw['requirements'][0]['paths']=['report.pdf','slides.pptx']
    assert TaskContract.model_validate(raw,context={'sources':[task]}).requirements[0].paths==['report.pdf','slides.pptx']
    raw['requirements'][0]['paths'].append('invented.pdf')
    with pytest.raises(ValidationError,match='exact named files'):
        TaskContract.model_validate(raw,context={'sources':[task]})


def test_recorded_flash_contract_is_rejected_for_omitted_deliverables():
    # Minimal reproduction of the real saved contract: it named two services,
    # but checked existence of only data_service.py.
    task='Implement data_service.py and report_service.py.'
    with pytest.raises(ValidationError,match='report_service.py'):
        TaskContract.model_validate({'requirements':[{'id':'r1','text':task,
            'validation':'artifact_exists','path':'data_service.py','source_quote':task}]},
            context={'sources':[task]})


def test_grouped_outputs_distinguish_readonly_input_from_deliverables():
    task='Use data.json to create report.pdf and slides.pptx.'
    raw={'requirements':[{'id':'r1','text':task,'validation':'artifact_exists',
        'path':'report.pdf','paths':['slides.pptx'],'inputs':['data.json'],'source_quote':task}]}
    result=TaskContract.model_validate(raw,context={'sources':[task]})
    assert result.requirements[0].inputs==['data.json']
    raw['requirements'][0]['inputs']=['unknown.json']
    with pytest.raises(ValidationError,match='exact named read-only sources'):
        TaskContract.model_validate(raw,context={'sources':[task]})
    raw['requirements'][0]['inputs']=['report.pdf']
    with pytest.raises(ValidationError,match='separate from output'):
        TaskContract.model_validate(raw,context={'sources':[task]})


def test_artifact_inspector_rejects_pdf_header_and_rechecks_replaced_file(tmp_path):
    import fitz
    ws=workspace(tmp_path);p=tmp_path/'report.pdf';p.write_bytes(b'%PDF-1.4')
    first=asyncio.run(inspect_artifact(ws,'report.pdf'))
    assert not first['ok'] and 'FileDataError' in first['error']
    with fitz.open() as doc:
        doc.new_page().insert_text((40,40),'Population report')
        doc.new_page().insert_text((40,40),'Source and method')
        doc.save(p)
    good=asyncio.run(inspect_artifact(ws,'report.pdf'))
    assert good['ok'] and good['pages']==2 and 'Source and method' in good['text']
    logs=len(list(ws.directory.glob('*.log')))
    assert asyncio.run(inspect_artifact(ws,'report.pdf'))==good
    assert len(list(ws.directory.glob('*.log')))==logs  # no repeated parser process
    p.unlink()
    assert not asyncio.run(inspect_artifact(ws,'report.pdf'))['ok']


def test_pptx_inspection_exposes_empty_cover_not_a_completed_deck(tmp_path):
    from pptx import Presentation
    prs=Presentation();prs.slides.add_slide(prs.slide_layouts[0]).shapes.title.text='Анализ населения'
    prs.save(tmp_path/'slides.pptx')
    result=asyncio.run(inspect_artifact(workspace(tmp_path),'slides.pptx'))
    assert result['ok'] and result['slides']==1 and result['charts']==0
    assert result['ratio']==pytest.approx(4/3) and 'Анализ населения' in result['text']


def test_unused_chart_part_does_not_count_as_delivered_chart(tmp_path):
    import zipfile
    from pptx import Presentation
    prs=Presentation();prs.slides.add_slide(prs.slide_layouts[0]);prs.save(tmp_path/'slides.pptx')
    with zipfile.ZipFile(tmp_path/'slides.pptx','a') as z:
        z.writestr('ppt/charts/chart999.xml','<unused/>')
    result=asyncio.run(inspect_artifact(workspace(tmp_path),'slides.pptx'))
    assert result['ok'] and result['charts']==0


def test_json_and_path_boundaries(tmp_path):
    ws=workspace(tmp_path);(tmp_path/'data.json').write_text('{')
    assert not asyncio.run(inspect_artifact(ws,'data.json'))['ok']
    with pytest.raises(ValueError):asyncio.run(inspect_artifact(ws,'../outside.pdf'))
    (tmp_path/'.env').write_text('PRIVATE')
    with pytest.raises(ValueError):asyncio.run(inspect_artifact(ws,'.env'))


def test_missing_output_blocks_before_expensive_critic_and_queue_prioritizes_work(tmp_path):
    async def forbidden(*args):raise AssertionError('No inference needed for absent output')
    ws=workspace(tmp_path);control=SessionControl()
    critic=ContextCritic(forbidden,Supervisor(ws.directory,control,forbidden,False),control,{'enabled':True})
    contract=TaskContract(requirements=[
        {'id':'r1','text':'Run tests','validation':'registered_checks'},
        {'id':'r2','text':'Create both files','validation':'artifact_exists','path':'build.py','paths':['report.pdf']}])
    (tmp_path/'build.py').write_text('# generator')
    graph=ws.obligations();graph.sync(contract,[])
    missing=asyncio.run(critic.missing_conditions(SimpleNamespace(id='main'),ws,contract))
    queue=agenda(contract,graph,missing,mode='verify',revision=0,steering_version=0)
    assert queue['focus']['requirement']=='r2' and queue['suggested_mode']=='implement'
    assert 'report.pdf' in queue['focus']['action'] and critic.calls==0


def test_repair_advice_survives_resume_but_not_task_change(tmp_path):
    ws=workspace(tmp_path);contract=TaskContract(requirements=[{'id':'r1','text':'Fix validation','validation':'model_evidence'}])
    graph=ws.obligations();graph.sync(contract,[])
    graph.review(Assessment(aligned=True,ready=False,criteria=[{'requirement':'r1','status':'missing','evidence':[]}],
        issues=['Wrong validation'],next_mode='repair',next_steps=[{'requirement':'r1','problem':'bool accepted as integer',
        'action':'Reject bool before int conversion','mode':'repair','verification':'bool raises TypeError; integers remain supported'}]),[])
    restored=ws.obligations()
    current=agenda(contract,restored,[],mode='verify',revision=0,steering_version=0)
    assert current['focus']['action']=='Reject bool before int conversion'
    after_edit=agenda(contract,restored,[],mode='verify',revision=1,steering_version=0)
    assert after_edit['focus']['status']=='recheck_after_change' and after_edit['suggested_mode']=='inspect'
    restored.sync(contract,[{'text':'Do not modify source; explain only'}])
    assert not restored.data.get('next_steps')


def test_concrete_repair_precedes_generic_completion_gap_and_final_check(tmp_path):
    ws=workspace(tmp_path)
    contract=TaskContract(requirements=[
        {'id':'r1','text':'Complete the product','validation':'model_evidence'},
        {'id':'r2','text':'Measure actual playback','validation':'model_evidence'},
        {'id':'r3','text':'Run tests after changes','validation':'registered_checks'}])
    graph=ws.obligations();graph.sync(contract,[])
    graph.review(Assessment(aligned=True,ready=False,criteria=[
        {'requirement':r.id,'status':'uncertain','evidence':[]} for r in contract.requirements],
        issues=['Paused media is counted as viewing'],next_mode='repair',next_steps=[{
            'requirement':'r2','problem':'Paused media is counted as viewing',
            'action':'Accumulate actual playback only','mode':'repair',
            'verification':'Paused playback increments no viewing time'}]),[])
    for missing in ([],[contract.requirements[2]]):
        queue=agenda(contract,graph,missing,mode='repair',revision=0,steering_version=0)
        assert queue['focus']['requirement']=='r2'
        assert queue['suggested_mode']=='repair'
        assert {r['requirement'] for r in queue['items']}=={'r1','r2','r3'}


def test_pending_question_is_not_assumed_and_survives_restart(tmp_path):
    ws=workspace(tmp_path);control=SessionControl();control.bind(ws.directory/'steering.jsonl')
    attention=Attention(ws,control)
    args={'question':'Which currency should the invoice use?','why':'The source has USD and EUR amounts','options':['USD','EUR']}
    result=asyncio.run(attention.ask(args))
    assert result['needs_input'] and not result['ok']
    assert clarify.ledger()[-1]['status']=='pending' and not clarify.ledger()[-1]['answer']
    restored=Attention(ws,control,resume=True)
    assert len(restored.pending())==1
    with pytest.raises(ValueError,match='newer real operator'):
        asyncio.run(restored.ask({**args,'answer_id':'invented'}))
    receipt=control.post('EUR',interrupt=False)
    resolved=asyncio.run(restored.ask({**args,'answer_id':receipt['id']}))
    assert resolved['ok'] and not restored.pending()
    assert json.loads(resolved['output'])['answer']=='EUR'


def test_actual_frontend_answer_is_posted_once_and_reused(tmp_path,monkeypatch):
    calls=[]
    def frontend(items):calls.append(items);return ['Berlin']
    monkeypatch.setattr(clarify,'_clarifier',frontend)
    control=SessionControl();ws=workspace(tmp_path);control.bind(ws.directory/'steering.jsonl')
    a=Attention(ws,control);args={'question':'Which city?','why':'The city changes the search'}
    assert asyncio.run(a.ask(args))['ok']
    assert asyncio.run(a.ask(args))['ok']
    assert len(calls)==len(control.history)==1 and 'Berlin' in control.history[0]['text']


def test_child_question_is_handed_to_parent_without_contacting_user(tmp_path,monkeypatch):
    def frontend(items):raise AssertionError('Child must not contact the user directly')
    monkeypatch.setattr(clarify,'_clarifier',frontend)
    ws=workspace(tmp_path);ws.owner_id='mind-1'
    result=asyncio.run(Attention(ws,SessionControl()).ask({'question':'Which API version?','why':'Contract differs'}))
    assert result['needs_input'] and result['human_action']['status']=='pending'


def test_readonly_child_copy_does_not_inherit_parent_attention_channel(tmp_path,monkeypatch):
    import copy
    from src.coding.worktools import execute_extra
    def frontend(items):raise AssertionError('Read-only child must hand off to parent')
    monkeypatch.setattr(clarify,'_clarifier',frontend)
    parent=workspace(tmp_path);control=SessionControl();parent.owner_id='main'
    parent.orchestrator=SimpleNamespace(control=control)
    parent.attention=Attention(parent,control)
    child=copy.copy(parent);child.owner_id='mind-1'
    result=asyncio.run(execute_extra(child,'ask_user',{'question':'Which API version?','why':'Contract differs'}))
    assert result['needs_input'] and child.attention.owner=='mind-1'
    assert not parent.attention.pending() and not control.snapshot.get('human_action')


def test_headless_agent_stops_with_question_instead_of_claiming_completion(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            return schema.model_validate({'action':'tool','tool':'ask_user','args':{
                'question':'Which account is the export for?','why':'Multiple accounts are present'}})
    result=asyncio.run(CodingHarness(tmp_path,model=Model(),config={'planning':'adaptive','max_turns':3}).run('Prepare an export'))
    assert result['status']=='paused' and result['completion']['state']=='needs_input'
    assert result['completion']['human_action'][0]['question'] in result['final_answer']
    assert not result['checks_passed']


def test_turn_limit_answer_explains_the_actual_failed_operation(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            return schema.model_validate({'action':'tool','tool':'check','args':{'name':'unit'}})
    result=asyncio.run(CodingHarness(tmp_path,model=Model(),config={
        'planning':'adaptive','max_turns':2,'learn':False,
        'checks':{'unit':[sys.executable,'-c','raise RuntimeError("dependency unavailable in check environment")']}
    }).run('Run the unit check'))
    assert result['status']=='paused'
    assert 'dependency unavailable in check environment' in result['final_answer']
    assert 'check' in result['final_answer'] and 'resume' in result['final_answer']


def test_critic_gap_becomes_repair_then_real_check_and_finish(tmp_path):
    (tmp_path/'value.py').write_text('VALUE=1\n')
    class Model:
        actor=0;reviews=0;packets=[]
        async def __call__(self,messages,schema,effort):
            data=json.loads(messages[-1]['content'])
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Set VALUE=2','validation':'model_evidence'}],
                    'domains':['software'],'response_format':'text','response_keys':[]})
            if 'oneOf' not in schema.model_json_schema():
                self.reviews+=1;ready=self.reviews>1
                return schema.model_validate({'aligned':True,'ready':ready,
                    'criteria':{'r1':{'status':'supported' if ready else 'missing','evidence':data['valid_evidence_ids'][-1:]}},
                    'issues':[] if ready else ['Required value is unavailable: VALUE is still 1'],
                    'next_mode':'verify' if ready else 'repair','next_steps':[] if ready else [{
                        'requirement':'r1','problem':'VALUE is still 1','action':'Set VALUE to 2 in value.py',
                        'mode':'repair','verification':'Import VALUE and assert it equals 2'}]})
            self.packets.append(data);self.actor+=1
            actions=[{'action':'finish','summary':'Done'},
                {'action':'tool','tool':'edit','args':{'path':'value.py','old':'VALUE=1','new':'VALUE=2'}},
                {'action':'tool','tool':'check','args':{'name':'unit'}},
                {'action':'finish','summary':'VALUE=2; unit passed.'}]
            return schema.model_validate(actions[self.actor-1])
    model=Model()
    h=CodingHarness(tmp_path,model=model,config={'planning':'adaptive','max_turns':5,'learn':False,
        'checks':{'unit':[sys.executable,'-B','-c','from value import VALUE; assert VALUE==2']},
        'orchestration':{'enabled':True,'auto_domains':True,'same_model':True,
                         'critic':{'enabled':True,'active':False,'max_calls':5}}})
    result=asyncio.run(h.run('Set VALUE=2'))
    assert result['status']=='done' and result['checks_passed']==['unit']
    assert result['completion']['remaining_work']['remaining']==0
    assert model.reviews==2 and (tmp_path/'value.py').read_text()=='VALUE=2\n'
    assert model.packets[1]['followthrough']['focus']['requirement']=='r1'
    assert model.packets[1]['cognitive_mode']=='repair'
    assert 'Set VALUE to 2' in model.packets[1]['next_action_hint']


def test_resume_input_resolves_pending_question_without_losing_original_task(tmp_path):
    class First:
        async def __call__(self,messages,schema,effort):
            return schema.model_validate({'action':'tool','tool':'ask_user','args':{
                'question':'Which currency?','why':'USD and EUR are both present'}})
    config={'planning':'adaptive','max_turns':3,'learn':False}
    first=asyncio.run(CodingHarness(tmp_path,model=First(),config=config).run('Summarize the invoice',session_id='currency'))
    assert first['completion']['state']=='needs_input'
    class Resume:
        calls=0
        async def __call__(self,messages,schema,effort):
            self.calls+=1;packet=json.loads(messages[-1]['content'])
            assert packet['task']=='Summarize the invoice'
            if self.calls==1:
                reply=next(x for x in packet['human_steering'] if x['text']=='EUR')
                return schema.model_validate({'action':'tool','tool':'ask_user','args':{
                    'question':'Which currency?','why':'USD and EUR are both present','answer_id':reply['id']}})
            assert not packet.get('human_action')
            return schema.model_validate({'action':'finish','summary':'Invoice summary in EUR.'})
    result=asyncio.run(CodingHarness(tmp_path,model=Resume(),config=config).run('EUR',session_id='currency',resume=True))
    assert result['status']=='done' and not result['completion']['human_action']


def test_finish_after_real_reply_gets_reconciliation_instead_of_reasking_user(tmp_path):
    class First:
        async def __call__(self,messages,schema,effort):
            return schema.model_validate({'action':'tool','tool':'ask_user','args':{
                'question':'Which currency?','why':'Currency is not specified'}})
    config={'planning':'adaptive','max_turns':4,'learn':False}
    asyncio.run(CodingHarness(tmp_path,model=First(),config=config).run('Summarize invoice',session_id='reply'))
    class Resume:
        calls=0
        async def __call__(self,messages,schema,effort):
            self.calls+=1;packet=json.loads(messages[-1]['content'])
            if self.calls==2:
                reply=next(x for x in packet['human_steering'] if x['text']=='EUR')
                assert 'answer_id' in str(packet['recent'])
                return schema.model_validate({'action':'tool','tool':'ask_user','args':{
                    'question':'Which currency?','why':'Currency is not specified','answer_id':reply['id']}})
            return schema.model_validate({'action':'finish','summary':'Invoice uses EUR.'})
    model=Resume()
    result=asyncio.run(CodingHarness(tmp_path,model=model,config=config).run('EUR',session_id='reply',resume=True))
    assert model.calls==3 and result['status']=='done'
    assert not result['completion']['human_action']


def test_finish_without_new_operator_reply_still_needs_input(tmp_path):
    class First:
        async def __call__(self,messages,schema,effort):
            return schema.model_validate({'action':'tool','tool':'ask_user','args':{
                'question':'Which currency?','why':'Currency is not specified'}})
    class Finish:
        async def __call__(self,messages,schema,effort):
            return schema.model_validate({'action':'finish','summary':'Assuming EUR.'})
    config={'planning':'adaptive','max_turns':3,'learn':False}
    asyncio.run(CodingHarness(tmp_path,model=First(),config=config).run('Summarize invoice',session_id='no-reply'))
    result=asyncio.run(CodingHarness(tmp_path,model=Finish(),config=config).run('',session_id='no-reply',resume=True))
    assert result['completion']['state']=='needs_input'
    assert result['completion']['human_action'][0]['status']=='pending'


def test_interrupted_write_is_not_replayed_blindly(tmp_path):
    action={'action':'tool','tool':'write','args':{'path':'receipt.txt','content':'once'}}
    class CrashingWorkspace(Workspace):
        async def execute(self,name,args,readonly=False):
            result=await super().execute(name,args,readonly)
            if name=='write' and result['ok']: raise asyncio.CancelledError()
            return result
    class First:
        async def __call__(self,messages,schema,effort):return schema.model_validate(action)
    config={'planning':'adaptive','max_turns':4,'learn':False,
            'checks':{'unit':[sys.executable,'-B','-c',"from pathlib import Path; assert Path('receipt.txt').read_text()=='once'"]}}
    h=CodingHarness(tmp_path,model=First(),config=config,workspace_factory=CrashingWorkspace)
    with pytest.raises(asyncio.CancelledError):asyncio.run(h.run('Write receipt.txt',session_id='recovery'))
    assert (tmp_path/'receipt.txt').read_text()=='once'
    state=json.loads((h.workspace.directory/'state.json').read_text())
    assert state['pending_action'] and state['pending_operation']['tool']=='write'
    class Resume:
        calls=0
        async def __call__(self,messages,schema,effort):
            self.calls+=1;packet=json.loads(messages[-1]['content'])
            if self.calls==1:
                assert packet['interrupted_operation']['tool']=='write'
                return schema.model_validate(action)
            if self.calls==2:
                assert 'unknown outcome' in str(packet['recent'])
                return schema.model_validate({'action':'tool','tool':'read','args':{'path':'receipt.txt'}})
            if self.calls==3:return schema.model_validate({'action':'tool','tool':'check','args':{'name':'unit'}})
            return schema.model_validate({'action':'finish','summary':'Receipt exists once and was checked.'})
    result=asyncio.run(CodingHarness(tmp_path,model=Resume(),config=config).run('',resume=True,session_id='recovery'))
    assert result['status']=='done' and result['tools_used']==['read','check']
