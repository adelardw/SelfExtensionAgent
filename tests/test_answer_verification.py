import asyncio
import json

import pytest
from pydantic import ValidationError

from src.coding.answer_verification import bind_answer,validate_oracle_program
from src.coding.critic import ContextCritic,TaskContract
from src.coding.control import SessionControl
from src.coding.harness import Workspace
from src.coding.minds import MindScheduler
from src.coding.supervisor import Supervisor


def test_binding_checks_witness_not_just_correct_objective_and_preserves_types():
    checks={'checks':[{'field':'distance','op':'equal','expected':3},
                     {'field':'path','op':'one_of','expected':[[0,2,3],[0,4,3]]}]}
    assert not bind_answer('{"distance":3,"path":[0,1,3]}',checks)['ok']
    assert bind_answer('{"distance":3,"path":[0,4,3]}',checks)['ok']
    assert not bind_answer('{"distance":3.0,"path":[0,4,3]}',checks)['ok']
    with pytest.raises(ValueError,match='omitted'):
        bind_answer('{"distance":3,"path":[0,2,3],"count":2}',checks)


def test_binding_exact_fractions_unordered_alternatives_and_malformed_oracles():
    assert bind_answer('{"value":"2/6"}',{'checks':[{'field':'value','op':'fraction','expected':'1/3'}]})['ok']
    checks={'checks':[{'field':'vertices','op':'set_one_of','expected':[['a','b'],['x','y']]}]}
    assert bind_answer('{"vertices":["b","a"]}',checks)['ok']
    assert not bind_answer('{"vertices":["b","a","a"]}',checks)['ok']
    with pytest.raises(ValueError):bind_answer('{"value":"1/3"}',{'checks':[]})
    with pytest.raises(ValueError):bind_answer('{"value":"1/3"}',{'checks':[{'field':'value','op':'fraction','expected':'1/0'}]})
    with pytest.raises(ValueError):bind_answer('{"value":42}',{'checks':[{'field':'value','op':'equal','expected':42}]},['different_key'])


def test_contract_cannot_rename_an_explicit_answer_key():
    data={'requirements':[{'id':'r1','text':'Answer','validation':'model_evidence'}],
          'response_format':'json','response_keys':['expected_value','derivation']}
    with pytest.raises(ValidationError,match='exact key names'):
        TaskContract.model_validate(data,context={'sources':['Return JSON with value and derivation.'],'ground_answer_keys':True})
    data['response_keys']=['value','derivation']
    assert TaskContract.model_validate(data,context={'sources':['Return JSON with value and derivation.'],'ground_answer_keys':True})


def test_software_contract_does_not_invent_user_json_from_action_envelope(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            wire=schema.model_json_schema()['properties']
            assert wire['response_format']['const']=='text' and wire['response_keys']['const']==[]
            return schema.model_validate({'requirements':[{'id':'r1','text':'Repair code','validation':'model_evidence'}],
                'domains':['software'],'response_format':'json','response_keys':['status']})
    control,model=SessionControl(),Model();ws=Workspace(tmp_path,tmp_path/'.sea',{})
    supervisor=Supervisor(tmp_path,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main'];node.task='Repair the parser and preserve the public API.'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'answer_verification':True})
    contract=asyncio.run(critic.contract(node))
    assert contract.response_format=='text' and not contract.response_keys and critic.calls==1


def test_operator_can_revoke_original_json_delivery_without_stale_key_constraint(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            assert 'const' not in schema.model_json_schema()['properties']['response_keys']
            return schema.model_validate({'requirements':[{'id':'r1','text':'Explain in prose','validation':'model_evidence'}],
                'domains':['mathematics'],'response_format':'text','response_keys':[]})
    control,model=SessionControl(),Model();ws=Workspace(tmp_path,tmp_path/'.sea',{})
    control.post('Change the final format: answer in ordinary prose, without a JSON object.')
    supervisor=Supervisor(tmp_path,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main'];node.task='Return JSON with value and derivation.'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'answer_verification':True})
    contract=asyncio.run(critic.contract(node))
    assert contract.response_format=='text' and not contract.response_keys


def test_oracle_must_export_its_computation_not_unrelated_literal_guesses():
    with pytest.raises(ValueError,match='computed'):
        validate_oracle_program('distance=sum([1,2]); print({"checks":[{"expected":-13}]})')
    with pytest.raises(ValueError,match='computed'):
        validate_oracle_program('distance=-13; print({"checks":[{"expected":distance}]})')
    validate_oracle_program('distance=sum([1,2]); print({"checks":[{"expected":distance}]})')
    with pytest.raises(ValueError,match='valid Python'):validate_oracle_program('if:')
    with pytest.raises(ValueError,match='candidate argument'):
        validate_oracle_program('verify("path",lambda value: True)')


@pytest.mark.parametrize('typed',[False,True])
def test_witness_predicate_rechecks_new_candidates_without_regenerating_oracle(tmp_path,typed):
    from src.coding.answer_verification import rebind_observation
    from src.coding.verifier import ProbePlan,run_probes
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    code='''edges={(0,1):1,(1,3):2,(0,2):2,(2,3):1,(0,3):8}
optimum=min(edges[0,1]+edges[1,3],edges[0,2]+edges[2,3],edges[0,3])
check('cost',optimum)
def valid(path):
    return (isinstance(path,list) and len(path)>1 and path[0]==0 and path[-1]==3
            and len(set(path))==len(path) and all(e in edges for e in zip(path,path[1:]))
            and sum(edges[e] for e in zip(path,path[1:]))==optimum)
verify('path',valid)
'''
    validate_oracle_program(code)
    ws=Workspace(tmp_path,tmp_path/'.sea',{});control=SessionControl()
    from types import SimpleNamespace
    supervisor=Supervisor(tmp_path,control,SimpleNamespace(),False);supervisor.register('critic','review','witness')
    args={'code':code}
    if typed:
        args={'code':code.replace("check('cost',optimum)\n",'').replace("verify('path',valid)\n",''),
              'outputs':{'cost':'optimum','path':'valid'}}
    plan=ProbePlan.model_validate({'probes':[{'tool':'answer','args':args,'purpose':'Independent optimum and witness validity'}],'limitation':''})
    first=asyncio.run(run_probes(ws,plan,['answer'],supervisor,'critic',control,
                                candidate='{"cost":3,"path":[0,1,3]}',answer_keys=['cost','path']))[0]
    assert first['ok'],first
    # Merely reusing the old successful boolean cannot approve a new candidate.
    assert not bind_answer('{"cost":3,"path":[0,2,3]}',first['answer_checks'])['ok']
    second=asyncio.run(rebind_observation(ws,first,'{"cost":3,"path":[0,2,3]}',
                                        ['cost','path'],supervisor,'critic',control))
    assert second['ok'] and second['id']!=first['id']
    bad=asyncio.run(rebind_observation(ws,first,'{"cost":3,"path":[0,3]}',
                                     ['cost','path'],supervisor,'critic',control))
    assert not bad['ok'] and bad['error_kind']=='counterexample'


def test_answer_program_treats_template_names_as_data(tmp_path):
    from src.coding.answer_verification import run_answer_program
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    result=asyncio.run(run_answer_program(ws,'check("CODE",sum([1,2]))',['CODE']))
    assert result['ok'] and json.loads(result['output'])['checks'][0]['field']=='CODE'


def test_explicit_json_key_syntax_is_copied_without_renaming():
    from src.coding.delivery import explicit_json_keys
    assert explicit_json_keys('Find result. Return JSON with keys distance (integer), path (list of [u,v]), and explanation.')==['distance','path','explanation']
    assert explicit_json_keys('Return a JSON final answer with keys value (p/q) and derivation (string).')==['value','derivation']
    assert explicit_json_keys('Верни JSON с ключами "сумма" и `пояснение`.')==['сумма','пояснение']
    assert explicit_json_keys('Parse JSON, then describe the output.') is None
    assert explicit_json_keys('Return JSON with whatever fields seem useful.') is None


def test_runtime_exports_calculated_variables_without_model_json_boilerplate(tmp_path):
    from src.coding.answer_verification import run_answer_program
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    result=asyncio.run(run_answer_program(ws,
        'from sympy import Rational\nanswer=Rational(1,3)+Rational(1,6)\nprint("debug",end="")\ncheck(FIELDS[0],answer,op="fraction")',['value']))
    assert result['ok'],result
    assert json.loads(result['output'])=={'checks':[{'field':'value','expected':'1/2','op':'fraction'}]}


def test_plain_python_export_maps_infer_exact_types_and_validate_witness(tmp_path):
    from src.coding.answer_verification import run_answer_program
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    code='''from sympy import Rational
value=Rational(1,3)+Rational(1,6)
EXPECTED={'value':value}
EXPECTED['vertices']={x for x in ['a','b']}
def valid(path):return path in [[0,1,3],[0,2,3]]
VALIDATORS={'path':valid}
'''
    answer='{"value":"1/2","vertices":["b","a"],"path":[0,2,3]}'
    result=asyncio.run(run_answer_program(ws,code,['value','vertices','path'],answer))
    assert result['ok'],result
    assert bind_answer(answer,result['output'])['ok']
    assert not bind_answer('{"value":"1/3","vertices":["b","a"],"path":[0,2,3]}',result['output'])['ok']
    with pytest.raises(ValueError,match='computed'):validate_oracle_program('EXPECTED={"value":42}')
    with pytest.raises(ValueError,match='candidate argument'):
        validate_oracle_program('VALIDATORS={"path":lambda x: True}')


def test_required_output_expressions_bind_computed_values_and_witness_functions(tmp_path):
    from src.coding.answer_verification import answer_program_schema,run_answer_program
    from pydantic import ValidationError
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    fields=['value','vertices','path']
    schema=answer_program_schema(fields)
    outputs=schema.model_json_schema()['properties']['outputs']
    assert outputs['required']==fields and outputs['additionalProperties'] is False
    program={'code':'from fractions import Fraction\nvalue=Fraction(1,2)\nvertices=set(range(2))\n'
        'def valid(path): return path in [[0,1,3],[0,2,3]]',
        'outputs':{'value':'value','vertices':'vertices','path':'valid'},'limitation':'Finite example'}
    assert schema.model_validate(program)
    for bad in ({'value':'value'},{**program['outputs'],'invented':'value'}):
        with pytest.raises(ValidationError,match='exactly'):schema.model_validate({**program,'outputs':bad})
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    # No user-written export calls; the typed program computes and the runtime binds.
    candidate=json.dumps({'value':'1/2','vertices':[1,0],'path':[0,2,3]})
    result=asyncio.run(run_answer_program(ws,program['code'],fields,candidate,program['outputs']))
    assert result['ok'],result
    assert bind_answer(candidate,result['output'],fields)['ok']
    assert not bind_answer(candidate.replace('[0, 2, 3]','[0, 1, 2]'),result['output'],fields)['ok']
    with pytest.raises(ValueError,match='computed'):
        asyncio.run(run_answer_program(ws,'print("PASS")',['value'],'{"value":42}',{'value':'42'}))
    with pytest.raises(ValueError,match='candidate argument'):
        asyncio.run(run_answer_program(ws,'def valid(x): return True',['path'],'{"path":[]}',{'path':'valid'}))


def test_structured_finish_keys_are_constrained_on_wire_and_in_runtime():
    from src.coding.harness import Turn
    from src.coding.orchestration import scoped_schema
    schema=scoped_schema(Turn,{'math'},False,False,answer_keys=['value','derivation'])
    finish=next(b for b in schema.model_json_schema()['oneOf'] if b['properties']['action']['const']=='finish')
    assert finish['properties']['summary']['required']==['value','derivation']
    assert finish['properties']['summary']['additionalProperties'] is False
    turn=schema.model_validate({'action':'finish','summary':{'value':'1/2','derivation':'exact'}})
    assert isinstance(turn.summary,str) and json.loads(turn.summary)['value']=='1/2'
    with pytest.raises(ValidationError):schema.model_validate({'action':'finish','summary':{'expected_value':'1/2','derivation':'exact'}})


@pytest.mark.parametrize('operator',[None,'low','medium'])
def test_blind_oracle_repairs_actual_exception_without_losing_it_in_wrapper_source(tmp_path,operator):
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    class Model:
        probes=0
        async def __call__(self,messages,schema,effort):
            data=json.loads(messages[-1]['content'])
            if schema.__name__=='TaskContract':return schema.model_validate({
                'requirements':[{'id':'r1','text':'Compute value','validation':'model_evidence'}],
                'domains':['mathematics'],'response_format':'json','response_keys':['value']})
            if schema.__name__=='AnswerProgram':
                self.probes+=1
                if self.probes==1:
                    assert effort=='low'
                    code='padding='+repr('x'*3000)+'\nfrom fractions import missing_symbol\ntotal=sum([1,2])'
                else:
                    assert effort==(operator or 'medium')
                    note=data['previous_oracle_error']
                    assert 'ImportError' in note and 'missing_symbol' in note and len(note)<=1800
                    assert 'xxxx' not in note
                    code='total=sum([1,2])'
                return schema.model_validate({'code':code,'outputs':{'value':'total'},'limitation':'Finite arithmetic'})
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[{
                'requirement':'r1','status':'supported','evidence':[data['active_verification'][0]['id']]}],
                'issues':[],'next_mode':'reason'})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model();ws=Workspace(tmp_path,directory,{})
    control.reasoning=operator
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main'];node.task='Compute value as JSON'
    node.memory.summary='{"value":3}'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'answer_verification':True,'max_calls':4})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert result.ready and model.probes==2 and critic.calls==4


def test_blind_oracles_reject_wrong_delivery_and_rebind_without_regeneration(tmp_path):
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    class Model:
        probes=0
        reviews=0
        async def __call__(self,messages,schema,effort):
            data=json.loads(messages[-1]['content'])
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Return shortest path','validation':'model_evidence'}],
                    'domains':['algorithms'],'response_format':'json','response_keys':['distance','path','explanation']})
            if schema.__name__=='AnswerProgram':
                self.probes+=1
                assert set(data)<= {'task','human_steering','acceptance_contract','current_source','verifier_tools','answer_fields','previous_oracle_programs'}
                assert 'candidate_answer' not in data and 'working_memory' not in data
                assert data['answer_fields']==['distance','path']
                return schema.model_validate({'code':
                    'distance=sum([1,2]);path=[0]+[x for x in range(1,4) if x!=1]\n',
                    'outputs':{'distance':'distance','path':'path'},'limitation':'Finite graph only'})
            self.reviews+=1
            assert data['active_verification'][0]['ok']
            # Old candidate text and comparisons must not anchor the revised review.
            assert not data['evidence']
            records={p.stem:json.loads(p.read_text()) for p in directory.glob('[0-9]*.json')}
            old_candidates={eid for eid,item in records.items() if item.get('kind')=='candidate_answer'
                            and eid!=data['candidate_answer']['id']}
            assert not old_candidates & set(data['valid_evidence_ids'])
            assert not any('context_assessment' in o for o in data['working_memory']['observations'])
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[{
                'requirement':'r1','status':'supported','evidence':[data['active_verification'][0]['id']]}],
                'issues':[],'next_mode':'reason'})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model();ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task='Return shortest distance and path, with explanation, as JSON.'
    node.memory.summary='{"distance":3,"path":[0,1,3],"explanation":"An informal argument"}'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'answer_verification':True,'max_calls':8})
    first=asyncio.run(critic.assess(node,ws,'completion'))
    assert not first.ready and model.reviews==0 and model.probes==2
    node.memory.summary='{"distance":3,"path":[0,2,3],"explanation":"A revised argument"}'
    node.memory.observations.append({'context_assessment':{'ready':False,'issues':['Old answer was wrong']}})
    second=asyncio.run(critic.assess(node,ws,'completion'))
    assert second.ready and model.probes==2 and model.reviews==1
    node.memory.summary='{"distance":3,"path":[0,1,3],"explanation":"An incorrect witness"}'
    third=asyncio.run(critic.assess(node,ws,'completion'))
    assert not third.ready and model.reviews==1


def test_oracle_printing_success_without_checks_cannot_validate_answer(tmp_path):
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':return schema.model_validate({
                'requirements':[{'id':'r1','text':'Exact value','validation':'model_evidence'}],
                'domains':['mathematics'],'response_format':'json','response_keys':['value']})
            assert schema.__name__=='AnswerProgram','Failed oracle must not buy a semantic approval'
            return schema.model_validate({'code':'print("PASS")','outputs':{'value':'42'},'limitation':''})
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),Model();ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main'];node.task='Compute value as JSON'
    node.memory.summary='{"value":42}'
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'answer_verification':True,'max_calls':4})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert not result.ready and critic.calls==3
