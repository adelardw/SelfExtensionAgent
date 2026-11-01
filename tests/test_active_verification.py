import asyncio
import json
from types import SimpleNamespace

import pytest

from src.coding.control import SessionControl
from src.coding.critic import ContextCritic, TaskContract
from src.coding.harness import Workspace
from src.coding.minds import MindScheduler
from src.coding.supervisor import Supervisor
from src.coding.verifier import isolated_probe


def setup(tmp_path, model):
    if type(model) is object: model=SimpleNamespace()
    directory=tmp_path/'.sea';directory.mkdir()
    ws=Workspace(tmp_path,directory,{})
    control=SessionControl()
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task='ceildiv must round upward for positive integers'
    node.memory.summary='Implemented and all tests pass'
    return ws,control,supervisor,node


@pytest.mark.parametrize('kind,replacement,expected',[
    ('tool_error','valid',True),('tool_error','foreign',False),
    ('counterexample','valid',False),('inconclusive','valid',False)])
def test_failed_verifier_tool_requires_explicit_valid_replacement(tmp_path,kind,replacement,expected):
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Six is twice three','validation':'model_evidence'}]})
            if schema.__name__=='ProbePlan':
                return schema.model_validate({'probes':[
                    {'tool':'math','args':{'op':'unknown','expr':'2*3'},'purpose':'Check product'},
                    {'tool':'probe','args':{'code':'assert 2*3==6; print(6)'},'purpose':'Independent replacement'}],
                    'limitation':'Finite arithmetic only'})
            data=json.loads(messages[-1]['content']);bad,good=data['active_verification']
            assert bad['error_kind']=='tool_error' and good['ok']
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[
                {'requirement':'r1: Six is twice three','status':'supported','evidence':[good['id']]}],
                'issues':[],'next_mode':'reason','probe_resolutions':[{'id':bad['id'],'kind':kind,
                'reason':'The first observation used an unsupported tool operation, replaced by an independent calculation.',
                'replacement_evidence':[good['id'] if replacement=='valid' else '999']}]})
    model=Model();ws,control,supervisor,node=setup(tmp_path,model)
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'max_calls':4})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert result.ready is expected


def test_active_verifier_discovers_wrong_solution_despite_passing_old_tests(tmp_path):
    (tmp_path/'app.py').write_text('def ceildiv(a,b): return a//b\n')
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Correct ceiling division','validation':'model_evidence'}]})
            if schema.__name__=='ProbePlan':
                return schema.model_validate({'probes':[{'tool':'probe','args':{
                    'code':'from app import ceildiv\nassert ceildiv(7,3)==3, "counterexample: 7/3 must round to 3"'},
                    'purpose':'Non-divisible positive inputs'}],'limitation':'Finite boundary test'})
            data=json.loads(messages[-1]['content'])
            probe=data['active_verification'][0]
            assert not probe['ok'] and 'counterexample' in probe['output']
            return schema.model_validate({'aligned':True,'ready':False,'criteria':[
                {'requirement':'r1','status':'missing','evidence':[probe['id']]}],
                'issues':['Incorrect rounding for non-divisible inputs'],'next_mode':'repair'})
    ws,control,supervisor,node=setup(tmp_path,Model())
    old=ws.record({'ok':True,'output':'1 passed (6/3=2)'})
    node.memory.evidence=[old];node.memory.files={'app.py':ws.file_digest('app.py')}
    ws.passed={'old'};ws.verified_revision=ws.revision
    critic=ContextCritic(Model(),supervisor,control,{'enabled':True,'active':True,'max_calls':4})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert not result.ready and result.next_mode=='repair'
    assert critic.calls==3  # contract, probe plan, assessment; no observer polling
    assert (tmp_path/'app.py').read_text()=='def ceildiv(a,b): return a//b\n'
    assert ws.obligations().packet()['requirements'][0]['status']=='missing'
    assert not list((ws.directory/'branches').glob('probe-*'))
    assert any(a['tool']=='probe' for a in supervisor.agents['critic-main']['actions'])


def test_probe_writes_only_disposable_copy_and_detects_mutation_during_run(tmp_path,monkeypatch):
    ws,_,_,_=setup(tmp_path,object())
    (tmp_path/'data.txt').write_text('original')
    result=asyncio.run(isolated_probe(ws,'from pathlib import Path; Path("data.txt").write_text("test fixture"); print("ok")'))
    assert result['ok'] and (tmp_path/'data.txt').read_text()=='original'
    real=Workspace.command
    async def race(staging,*args,**kwargs):
        result=await real(staging,*args,**kwargs)
        (tmp_path/'data.txt').write_text('operator update')
        return result
    monkeypatch.setattr(Workspace,'command',race)
    result=asyncio.run(isolated_probe(ws,'print(42)'))
    assert not result['ok'] and result['stale']


def test_verifier_cannot_silently_ignore_failed_observation(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Correct','validation':'model_evidence'}]})
            if schema.__name__=='ProbePlan':
                return schema.model_validate({'probes':[{'tool':'probe','args':{'code':'assert False, "bad"'},'purpose':'Adversarial'}],'limitation':''})
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[
                {'requirement':'r1','status':'supported','evidence':['1']}],'issues':[],'next_mode':'review'})
    ws,control,supervisor,node=setup(tmp_path,Model())
    node.memory.evidence=[ws.record({'ok':True,'output':'agent says correct'})]
    critic=ContextCritic(Model(),supervisor,control,{'enabled':True,'active':True})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert not result.ready and any('Unaddressed' in x for x in result.issues)


def test_active_budget_reserves_assessment_and_remote_ceiling(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            assert schema.__name__=='TaskContract'
            return schema.model_validate({'requirements':[{'id':'r1','text':'Correct','validation':'model_evidence'}]})
    ws,control,supervisor,node=setup(tmp_path,Model())
    critic=ContextCritic(Model(),supervisor,control,{'enabled':True,'active':True,'max_calls':2})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert not result.ready and critic.calls==1 and 'budget' in result.issues[0]


def test_claim_dag_rejects_cycles_and_invalidates_evidence_and_dependents(tmp_path):
    ws,_,_,_=setup(tmp_path,object())
    (tmp_path/'app.py').write_text('VALUE=1')
    graph=ws.obligations()
    graph.sync(TaskContract(requirements=[{'id':'r1','text':'Correct','validation':'model_evidence'}]),[])
    eid=ws.record({'ok':True,'output':'VALUE=1'})
    graph.propose(id='c-base',text='Value is 1',requirements=['r1'],files=['app.py'],evidence=[eid])
    graph.propose(id='c-derived',text='Depends on base',depends_on=['c-base'])
    with pytest.raises(ValueError,match='Cyclic'):
        graph.propose(id='c-base',text='cycle',depends_on=['c-derived'])
    (tmp_path/'app.py').write_text('VALUE=2')
    statuses={c['id']:c['status'] for c in graph.packet()['claims']}
    assert statuses=={'c-base':'stale','c-derived':'stale'}
    (tmp_path/'app.py').write_text('VALUE=1')
    (ws.directory/(eid+'.json')).write_text('{}')
    assert all(c['status']=='stale' for c in graph.packet()['claims'])
    with pytest.raises(ValueError): graph.propose(id='c-new',text='fake',evidence=['../../secret'])


def test_claim_updates_and_new_requirements_invalidate_old_support(tmp_path):
    ws,_,_,_=setup(tmp_path,object())
    graph=ws.obligations()
    contract=TaskContract(requirements=[{'id':'r1','text':'A','validation':'model_evidence'}])
    graph.sync(contract,[])
    graph.propose(id='c-a',text='A')
    graph.propose(id='c-b',text='B',depends_on=['c-a'])
    graph.propose(id='c-a',text='Actually C')
    assert graph.status(graph.data['claims']['c-b'])=='stale'
    graph.sync(contract,[{'text':'new operator constraint'}])
    assert all(c['status']=='stale' for c in graph.packet()['claims'])


def test_answer_repair_reuses_current_observation_but_source_change_invalidates_it(tmp_path):
    class Model:
        plans=0
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Correct','validation':'model_evidence'}]})
            if schema.__name__=='ProbePlan':
                self.plans+=1
                return schema.model_validate({'probes':[{'tool':'probe','args':{'code':'print(2+2)'},'purpose':'Independent computation'}],'limitation':'Finite calculation'})
            data=json.loads(messages[-1]['content'])
            assert data['active_verification'][0]['args']['code']=='print(2+2)'
            ready=data['candidate_answer']['text']=='4'
            return schema.model_validate({'aligned':True,'ready':ready,'criteria':[
                {'requirement':'r1','status':'supported' if ready else 'missing','evidence':[data['active_verification'][0]['id']]}],
                'issues':[],'next_mode':'reason'})
    model=Model();ws,control,supervisor,node=setup(tmp_path,model)
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'max_calls':6})
    node.memory.summary='5'
    assert not asyncio.run(critic.assess(node,ws,'finish')).ready
    node.memory.summary='4'
    assert asyncio.run(critic.assess(node,ws,'finish')).ready
    assert critic.calls==4 and model.plans==1
    (tmp_path/'new.py').write_text('CHANGED=True')
    assert ws.obligations().packet()['requirements'][0]['status']=='stale'
    asyncio.run(critic.assess(node,ws,'finish'))
    assert model.plans==2


def test_known_log_alias_is_resolved_but_foreign_evidence_stays_rejected(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'A','validation':'model_evidence'}]})
            assert '1' in schema.model_json_schema()['$defs']['Criterion']['properties']['evidence']['items']['enum']
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[
                {'requirement':'r1','status':'supported','evidence':['real-log-id']}],'issues':[],'next_mode':'review'})
    ws,control,supervisor,node=setup(tmp_path,Model())
    node.memory.evidence=[ws.record({'ok':True,'output':'observed','log_id':'real-log-id'})]
    result=asyncio.run(ContextCritic(Model(),supervisor,control,{'enabled':True}).assess(node,ws,'finish'))
    assert result.ready and result.criteria[0].evidence==['1']
