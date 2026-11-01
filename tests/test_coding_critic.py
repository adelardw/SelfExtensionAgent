"""Semantic contracts cannot silently omit requested tests, worker use or evidence."""
import asyncio
import json
import pytest

from src.coding.critic import ContextCritic
from src.coding.control import SessionControl
from src.coding.harness import Workspace
from src.coding.supervisor import Supervisor
from src.coding.minds import MindScheduler


def test_contract_requires_explicit_domain_and_delivery_decisions_from_the_model():
    from src.coding.critic import TaskContract
    schema=TaskContract.model_json_schema()
    assert set(schema['required'])=={'requirements','domains','response_format','response_keys'}
    assert all('default' not in field for field in schema['properties'].values())


@pytest.mark.parametrize('status,evidence,issues,expected',[
    ('supported','current',[],True), ('uncertain','current',[],False),
    ('supported','foreign',[],False), ('supported','current',['Concrete missing boundary argument'],False),
])
def test_malformed_optional_strategy_cannot_erase_review_or_bypass_its_gates(tmp_path,status,evidence,issues,expected):
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Explain the result','validation':'model_evidence'}]})
            packet=json.loads(messages[-1]['content'])
            eid=packet['candidate_answer']['id'] if evidence=='current' else '999999'
            return schema.model_validate({'aligned':True,'ready':True,'criteria':{
                'r1':{'status':status,'evidence':[eid]}},'issues':issues,'next_mode':'verify',
                'profile':{'name':'new-review','base':'review','strategy':'x'*1201,
                           'tools':['invented_function'],'exit_when':['done']}})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model();ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task='Explain the result';node.memory.summary='An explanation'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':False,'max_calls':2})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert result.ready is expected and result.profile is None and critic.calls==2
    assert 'profile-rejected' in (directory/'supervisor.jsonl').read_text()


def test_bad_strategy_does_not_suppress_invalid_verification_fields():
    from src.coding.critic import Assessment
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Assessment.model_validate({'aligned':True,'ready':True,'criteria':'not a review',
            'issues':[],'next_mode':'verify','profile':{'tools':['invented_function']}})


@pytest.mark.parametrize('limit',[2,3])
def test_review_repairs_recorded_null_mode_once_without_relaxing_criteria(tmp_path,limit):
    from pydantic import ValidationError
    class Model:
        reviews=0
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Explain the result','validation':'model_evidence'}]})
            self.reviews+=1
            packet=json.loads(messages[-1]['content'])
            wire=schema.model_json_schema()['properties']['criteria']
            assert wire['type']=='object' and wire['required']==['r1'] and not wire['additionalProperties']
            if self.reviews==1:
                # Recorded live malformed review: ready=true, no criteria, null mode.
                return schema.model_validate({'aligned':True,'ready':True,'criteria':[],
                    'issues':[],'next_mode':None,'profile':None,'probe_resolutions':[]})
            assert 'next_mode' in messages[0]['content'] and 'Previous review did not validate' in messages[0]['content']
            return schema.model_validate({'aligned':True,'ready':True,'criteria':{
                'r1':{'status':'supported','evidence':[packet['candidate_answer']['id']]}},
                'issues':[],'next_mode':'verify'})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model();ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task='Explain the result';node.memory.summary='An explanation'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':False,'max_calls':limit})
    if limit==2:
        with pytest.raises(ValidationError):asyncio.run(critic.assess(node,ws,'completion'))
        assert not critic.cache
    else:
        result=asyncio.run(critic.assess(node,ws,'completion'))
        assert result.ready and result.criteria[0].requirement=='r1: Explain the result'
    assert critic.calls==limit and model.reviews==limit-1


@pytest.mark.parametrize('criteria,raises',[
    ({},True),({'foreign':{'status':'supported','evidence':['1']}},True),
    ({'r1':{'requirement':'r2','status':'supported','evidence':['1']}},True),
    ({'r1':{'status':'supported','evidence':['999']}},False),
    ({'r1':{'status':'supported','evidence':[]}},False),
])
def test_keyed_review_cannot_skip_requirements_or_invent_evidence(tmp_path,criteria,raises):
    from pydantic import ValidationError
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Explain','validation':'model_evidence'}]})
            return schema.model_validate({'aligned':True,'ready':True,'criteria':criteria,'issues':[],'next_mode':'verify'})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model();ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main'];node.task='Explain';node.memory.summary='Candidate'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':False,'max_calls':3})
    if raises:
        with pytest.raises(ValidationError):asyncio.run(critic.assess(node,ws,'completion'))
        assert critic.calls==3 and not critic.cache
    else:
        result=asyncio.run(critic.assess(node,ws,'completion'))
        assert not result.ready and result.criteria[0].status=='uncertain' and critic.calls==2


@pytest.mark.parametrize('issues,expected',[
    (['None'],True),([],True),(['The claimed complexity ignores the independent input dimension.'],False)])
def test_ready_flag_cannot_override_unresolved_material_issues(tmp_path,issues,expected):
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Justify the complexity',
                    'validation':'model_evidence'}],'domains':['algorithms']})
            data=json.loads(messages[-1]['content'])
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[{
                'requirement':'r1','status':'supported','evidence':[data['candidate_answer']['id']]}],
                'issues':issues,'next_mode':'reason'})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model();ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task='Justify the complexity';node.memory.summary='A proposed argument'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':False})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert result.ready is expected
    assert bool(result.issues) is (not expected)


def test_new_mechanical_contract_requires_an_actual_user_quote():
    from src.coding.critic import TaskContract
    from pydantic import ValidationError
    data={'requirements':[{'id':'r1','text':'Add tests','validation':'new_or_updated_tests'}]}
    with pytest.raises(ValidationError,match='source_quote'):
        TaskContract.model_validate(data,context={'sources':['Review the function']})
    data['requirements'][0]['source_quote']='Add tests'
    with pytest.raises(ValidationError,match='source_quote'):
        TaskContract.model_validate(data,context={'sources':['Review the function']})
    result=TaskContract.model_validate(data,context={'sources':['Add tests for empty input']})
    assert result.requirements[0].source_quote=='Add tests'
    # Historical checkpoint compatibility does not silently re-extract new obligations.
    assert TaskContract.model_validate(data).requirements


@pytest.mark.parametrize('limit',[1,2])
def test_contract_repairs_invalid_schema_once_within_shared_call_budget(tmp_path,limit):
    from pydantic import ValidationError
    class Model:
        calls=0
        async def __call__(self,messages,schema,effort):
            self.calls+=1
            if self.calls==2: assert 'response_format' in messages[0]['content']
            return schema.model_validate({'requirements':[
                {'id':'r1','text':'Return exact answer','validation':'model_evidence'}],
                'response_format':'' if self.calls==1 else 'json','response_keys':['value']})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model()
    ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main'];node.task='Return exact answer as JSON'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'max_calls':limit})
    if limit==1:
        with pytest.raises(ValidationError): asyncio.run(critic.contract(node))
        assert not critic.contracts
    else:
        result=asyncio.run(critic.contract(node))
        assert result.response_format=='json' and result.response_keys==['value']
        assert asyncio.run(critic.contract(node)) is result
    assert critic.calls==model.calls==limit


@pytest.mark.parametrize('answer',['The answer is 42.','{"wrong_key":42}'])
def test_final_json_contract_is_checked_before_spending_on_a_critic(tmp_path,answer):
    class Model:
        async def __call__(self,messages,schema,effort):
            assert schema.__name__=='TaskContract','Invalid delivery must not spend another review call'
            return schema.model_validate({'requirements':[
                {'id':'r1','text':'Return the value in JSON','validation':'model_evidence'}],
                'response_format':'json','response_keys':['value']})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model()
    ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    scheduler=MindScheduler(control,supervisor,{},ws)
    node=scheduler.nodes['main'];node.task='Return value as JSON';node.memory.summary=answer
    critic=ContextCritic(model,supervisor,control,{'enabled':True})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert not result.ready and 'valid JSON' in result.issues[0] and critic.calls==1


def test_critic_rejects_ready_claim_when_contract_artifact_is_missing(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[
                    {'id':'r1','text':'Add regression tests','validation':'new_or_updated_tests','source_quote':'Add regression tests'},
                    {'id':'r2','text':'Run registered checks','validation':'registered_checks','source_quote':'run checks'}]})
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[
                {'requirement':'r1: Add regression tests','status':'supported','evidence':['1']},
                {'requirement':'r2: Run checks','status':'supported','evidence':['1']}],
                'issues':[],'next_mode':'verify'})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model()
    ws=Workspace(tmp_path,directory,{})
    ws.record({'ok':True,'output':'old checks passed'})
    ws.passed={'unit'};ws.verified_revision=ws.revision
    supervisor=Supervisor(directory,control,model,False)
    scheduler=MindScheduler(control,supervisor,{},ws)
    node=scheduler.nodes['main'];node.task='Add regression tests and run checks';node.memory.evidence=['1']
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'max_calls':4})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert not result.ready and result.criteria[0].status=='missing'
    assert result.next_mode=='implement'
    assert critic.calls==1, 'Missing tests must be caught before buying probe plans/reviews'
    (tmp_path/'test_regression.py').write_text('def test_regression(): pass\n')
    ws.changed.add('test_regression.py')
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert result.ready and critic.calls==2, 'Satisfying the cheap gate must still require semantic review'


@pytest.mark.parametrize('validation,path',[
    ('artifact_exists','missing.txt'),('registered_checks',''),
    ('adopted_worker_changes',''),('new_or_updated_tests','')])
def test_mechanical_missing_conditions_do_not_spend_on_active_review(tmp_path,validation,path):
    class Model:
        async def __call__(self,messages,schema,effort):
            assert schema.__name__=='TaskContract'
            return schema.model_validate({'requirements':[{
                'id':'r1','text':'Explicit requirement','validation':validation,'path':path,'source_quote':'Explicit requirement'}]})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model()
    ws=Workspace(tmp_path,directory,{})
    # A deleted test and stale checks are not current satisfaction of the obligation.
    ws.changed.add('test_deleted.py');ws.passed={'unit'};ws.verified_revision=-1
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main'];node.task='Explicit requirement '+path
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert not result.ready and result.criteria[0].status=='missing' and critic.calls==1
    assert result.next_mode==('verify' if validation=='registered_checks' else 'implement')


def test_critic_requires_every_contract_id_and_rejects_invented_evidence(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[
                    {'id':'r1','text':'A','validation':'model_evidence'},
                    {'id':'r2','text':'B','validation':'model_evidence'}]})
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[
                {'requirement':'r1: A','status':'supported','evidence':['999']}],
                'issues':[],'next_mode':'review'})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model()
    ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    scheduler=MindScheduler(control,supervisor,{},ws)
    node=scheduler.nodes['main'];node.task='A and B'
    result=asyncio.run(ContextCritic(model,supervisor,control,{'enabled':True}).assess(node,ws,'completion'))
    assert not result.ready
    assert len(result.criteria)==2 and all(c.status=='uncertain' for c in result.criteria)
