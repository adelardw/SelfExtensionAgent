"""A fallible executable oracle may be challenged, never silently discarded."""
import asyncio
import json

import pytest

from src.coding.critic import ContextCritic
from src.coding.control import SessionControl
from src.coding.harness import Workspace
from src.coding.minds import MindScheduler
from src.coding.supervisor import Supervisor


# Reproduce the live fault: Bellman-Ford repeatedly counts the same shortest path.
BAD_COUNT='''edges=[(0,1,-5),(1,2,-5),(0,2,7)]
dist=[0,float('inf'),float('inf')];count=[1,0,0]
for _ in range(2):
    for u,v,w in edges:
        if dist[u]+w<dist[v]:dist[v]=dist[u]+w;count[v]=count[u]
        elif dist[u]+w==dist[v]:count[v]+=count[u]
distance=dist[2];total=count[2]
'''
ENUMERATE='''edges=[(0,1,-5),(1,2,-5),(0,2,7)]
paths=[]
def visit(u,cost):
    if u==2:paths.append(cost);return
    for a,b,w in edges:
        if a==u:visit(b,cost+w)
visit(0,0)
distance=min(paths);total=sum(value==distance for value in paths)
'''


def setup(tmp_path,*,candidate=1,resolution='oracle_error',first=BAD_COUNT,alternative=ENUMERATE,limit=8):
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    class Model:
        programs=0
        reviews=0
        async def __call__(self,messages,schema,effort):
            data=json.loads(messages[-1]['content'])
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{
                    'id':'r1','text':'Shortest distance and number of shortest paths','validation':'model_evidence'}],
                    'domains':['algorithms'],'response_format':'json','response_keys':['distance','count']})
            if schema.__name__=='AnswerProgram':
                self.programs+=1
                assert 'candidate_answer' not in data and 'working_memory' not in data
                assert 'active_verification' not in data and 'evidence' not in data
                if 'previous_oracle_programs' in data:
                    assert effort=='medium'
                    assert 'DIFFERENT justified method' in messages[0]['content']
                    assert data['previous_oracle_programs'][0]['code']==first
                else:assert effort=='low'
                code=alternative if 'previous_oracle_programs' in data else first
                return schema.model_validate({'code':code,'outputs':{'distance':'distance','count':'total'},
                                              'limitation':'Finite DAG enumeration'})
            self.reviews+=1
            wire=schema.model_json_schema()
            assert wire['$defs']['Criterion']['properties']['requirement']['enum']==['r1']
            assert wire['properties']['issues']['items']['maxLength']==500
            bad,good=data['active_verification']
            assert not bad['ok'] and good['ok']
            assert set(data['oracle_dispute']['original'])=={bad['id']}
            assert set(data['oracle_dispute']['alternative'])=={good['id']}
            resolutions=[] if resolution=='omit' else [{'id':bad['id'],
                'kind':resolution if resolution!='foreign' else 'oracle_error',
                'reason':'Repeated Bellman-Ford relaxation adds the same path count again; enumerating each DAG path once gives one minimum.',
                'replacement_evidence':['999'] if resolution=='foreign' else [good['id']]}]
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[{
                'requirement':'r1','status':'supported','evidence':[good['id']]}],
                'issues':[],'next_mode':'reason','probe_resolutions':resolutions})
    directory=tmp_path/'.sea';directory.mkdir()
    model,control=Model(),SessionControl();ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task='Find shortest distance and path count in DAG [(0,1,-5),(1,2,-5),(0,2,7)] from 0 to 2. Return JSON with distance and count.'
    node.memory.summary=json.dumps({'distance':-10,'count':candidate})
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'answer_verification':True,'max_calls':limit})
    return model,critic,node,ws


@pytest.mark.parametrize('resolution,ready',[
    ('oracle_error',True),('omit',False),('foreign',False),('tool_error',False),('inconclusive',False)])
def test_correct_candidate_needs_explicit_grounded_resolution_of_wrong_oracle(tmp_path,resolution,ready):
    model,critic,node,ws=setup(tmp_path,resolution=resolution)
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert result.ready is ready
    assert (model.programs,model.reviews,critic.calls)==(2,1,4)
    packet=ws.obligations().packet()['verification']
    assert len(packet['observations'])==2 and not packet['observations'][0]['ok']
    assert packet['observations'][1]['ok'] and packet['dispute']
    assert len(packet['resolutions'])==(resolution!='omit')


@pytest.mark.parametrize('first,alternative',[(BAD_COUNT,ENUMERATE),(ENUMERATE,ENUMERATE),(BAD_COUNT,BAD_COUNT)])
def test_no_semantic_vote_can_accept_candidate_rejected_by_both_oracles(tmp_path,first,alternative):
    model,critic,node,ws=setup(tmp_path,candidate=2,first=first,alternative=alternative)
    for _ in range(2):
        result=asyncio.run(critic.assess(node,ws,'completion'))
        assert not result.ready
    assert (model.programs,model.reviews,critic.calls)==(2,0,3)
    assert len(ws.obligations().packet()['verification']['observations'])==2


def test_dispute_rebinds_evidence_without_regeneration_and_source_change_invalidates_it(tmp_path):
    model,critic,node,ws=setup(tmp_path)
    assert asyncio.run(critic.assess(node,ws,'completion')).ready
    first=ws.obligations().packet()['verification']['dispute']
    # Formatting changes the delivered candidate identity but not its values.
    node.memory.summary='{"count":1,"distance":-10}'
    assert asyncio.run(critic.assess(node,ws,'completion')).ready
    second=ws.obligations().packet()['verification']['dispute']
    assert first!=second and (model.programs,model.reviews,critic.calls)==(2,2,5)
    assert all(e!=old for group in second for e,old in zip(second[group],first[group]))
    (ws.root/'input.txt').write_text('new source snapshot')
    assert asyncio.run(critic.assess(node,ws,'completion')).ready
    assert (model.programs,model.reviews,critic.calls)==(4,3,8)


def test_dispute_reserves_review_budget_and_reuses_initial_calculation(tmp_path):
    model,critic,node,ws=setup(tmp_path,limit=3)
    assert not asyncio.run(critic.assess(node,ws,'completion')).ready
    assert (model.programs,model.reviews,critic.calls)==(1,0,2)
    # One more call permits an alternative AND its mandatory semantic review.
    critic.config['max_calls']=4
    assert asyncio.run(critic.assess(node,ws,'completion')).ready
    assert critic.calls==4


def test_alternative_execution_failure_does_not_drop_original_counterexample(tmp_path):
    model,critic,node,ws=setup(tmp_path,limit=4,alternative='raise ValueError("broken oracle")\ndistance=sum([1]);total=sum([1])')
    assert not asyncio.run(critic.assess(node,ws,'completion')).ready
    observations=ws.obligations().packet()['verification']['observations']
    assert len(observations)==2
    assert observations[0]['error_kind']=='counterexample'
    assert observations[1]['error_kind']=='tool_error'
    assert model.reviews==0 and critic.calls==3


@pytest.mark.parametrize('project',[False,True])
def test_domain_label_cannot_skip_answer_verification_or_replace_project_behavior_checks(tmp_path,project):
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    if project:(tmp_path/'app.py').write_text('def twice(x): return x\n')
    class Model:
        async def __call__(self,messages,schema,effort):
            data=json.loads(messages[-1]['content'])
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Compute twice three correctly',
                    'validation':'model_evidence'}],'domains':['algorithms','mathematics','software'],
                    'response_format':'json','response_keys':['value']})
            if schema.__name__ in {'ProbePlan','AnswerProgram'}:
                if project:
                    assert schema.__name__=='ProbePlan','A numeric report must not replace validation of project behavior'
                    return schema.model_validate({'probes':[{'tool':'probe','args':{
                        'code':'from app import twice\nassert twice(3)==6'},'purpose':'Verify project implementation'}],
                        'limitation':'Finite boundary'})
                assert schema.__name__=='AnswerProgram','Mixed domain tags must not disable answer binding in an empty workspace'
                return schema.model_validate({'code':'value=2*3','outputs':{'value':'value'},'limitation':''})
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[{
                'requirement':'r1','status':'supported','evidence':[data['active_verification'][0]['id']]}],
                'issues':[],'next_mode':'reason'})
    directory=tmp_path/'.sea';directory.mkdir()
    model,control=Model(),SessionControl();ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task='Compute twice three correctly and return JSON with value.'
    node.memory.summary='{"value":6}'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'answer_verification':True,'max_calls':4})
    assert asyncio.run(critic.assess(node,ws,'completion')).ready is (not project)
