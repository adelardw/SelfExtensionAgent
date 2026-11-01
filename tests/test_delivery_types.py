"""Explicit answer representation must not consume another mathematical oracle."""
import asyncio
import json

import pytest
from pydantic import ValidationError

from src.coding.delivery import explicit_json_fields,delivery_fields,validate_delivery_fields
from src.coding.harness import Turn,Workspace
from src.coding.orchestration import scoped_schema
from src.coding.critic import ContextCritic
from src.coding.control import SessionControl
from src.coding.supervisor import Supervisor
from src.coding.minds import MindScheduler


TASK='Compute the conditional expectation. Return JSON with value (reduced fraction p/q) and derivation (string).'


def test_explicit_objects_and_nested_file_mapping_are_valid_without_coercion():
    assert explicit_json_fields('Return JSON with result (object).')=={'result':{'type':'object'}}
    fields={'files':{'type':'object','additionalProperties':{'type':'string'},'minProperties':1,'maxProperties':2}}
    schema=scoped_schema(Turn,set(),False,False,answer_keys=['files'],answer_fields=fields)
    turn=schema.model_validate({'action':'finish','summary':{'files':{'module.py':'VALUE=2\n'}}})
    assert json.loads(turn.summary)['files']=={'module.py':'VALUE=2\n'}
    for invalid in ({}, {'a':3}, {'a':'a','b':'b','c':'c'}, 'not an object', ['a']):
        with pytest.raises(ValueError):validate_delivery_fields({'files':invalid},fields)


def test_nested_object_required_and_unknown_properties_remain_enforced():
    fields={'result':{'type':'object','properties':{'count':{'type':'integer'}},
                     'required':['count'],'additionalProperties':False}}
    validate_delivery_fields({'result':{'count':2}},fields)
    for invalid in ({}, {'count':True}, {'count':2,'extra':'not requested'}):
        with pytest.raises(ValueError):validate_delivery_fields({'result':invalid},fields)


def test_explicit_types_are_compiled_independently_of_field_names():
    fields=explicit_json_fields('Return JSON with chance (reduced fraction p/q), count (integer), '
        'witness (list of integers), valid (boolean), explanation (string), and unspecified.')
    assert {k:v.get('type') for k,v in fields.items()}=={
        'chance':'string','count':'integer','witness':'array','valid':'boolean','explanation':'string','unspecified':None}
    validate_delivery_fields({'chance':'2/3','count':2,'witness':[1,3],'valid':True,'explanation':'Proof'},fields)
    assert explicit_json_fields('Return JSON with whatever fields you like.') is None
    assert explicit_json_fields('Return JSON with value and derivation.')=={'value':{},'derivation':{}}
    assert explicit_json_fields('Верни JSON с ключами вероятность (несократимая дробь p/q) и пояснение (строка).')['вероятность']['type']=='string'


@pytest.mark.parametrize('value',[
    {'p':55,'q':37},55/37,'110/74','55/0','not a fraction',
])
def test_wrong_fraction_representation_is_rejected_before_finish(value):
    fields=explicit_json_fields(TASK)
    schema=scoped_schema(Turn,{'math'},answer_keys=['value','derivation'],answer_fields=fields)
    with pytest.raises(ValidationError):
        schema.model_validate({'action':'finish','summary':{'value':value,'derivation':'Computed exactly.'}})
    valid=schema.model_validate({'action':'finish','summary':{'value':'55/37','derivation':'Computed exactly.'}})
    assert json.loads(valid.summary)['value']=='55/37'
    finish=next(b for b in schema.model_json_schema()['oneOf'] if b['properties']['action']['const']=='finish')
    assert finish['properties']['summary']['properties']['value']['pattern']==fields['value']['pattern']


@pytest.mark.parametrize('actual,description',[(True,'integer'),([1,True],'list of integers'),(float('inf'),'number')])
def test_other_primitive_formats_do_not_accept_python_coercions(actual,description):
    fields=explicit_json_fields(f'Return JSON with result ({description}).')
    with pytest.raises(ValueError):validate_delivery_fields({'result':actual},fields)


def test_steering_replaces_or_removes_stale_type_constraints():
    keys=['value','derivation']
    assert delivery_fields(TASK,[],keys)['value']['type']=='string'
    replacement='Return JSON with value (integer) and derivation (string).'
    assert delivery_fields(TASK,[{'text':replacement}],keys)['value']['type']=='integer'
    assert delivery_fields(TASK,[{'text':'Instead put a numerator/denominator object in value.'}],keys)=={}


def test_malformed_numeric_delivery_does_not_buy_a_probe_or_semantic_review(tmp_path):
    class Model:
        calls=0
        async def __call__(self,messages,schema,effort):
            self.calls+=1
            assert schema.__name__=='TaskContract'
            return schema.model_validate({'requirements':[{'id':'r1','text':'Compute exact expectation',
                'validation':'model_evidence'}],'domains':['mathematics'],'response_format':'json',
                'response_keys':['value','derivation']})
    control,model=SessionControl(),Model();supervisor=Supervisor(tmp_path,control,model,False)
    ws=Workspace(tmp_path,tmp_path,{})
    node=MindScheduler(control,supervisor,{},ws).nodes['main'];node.task=TASK
    node.memory.summary=json.dumps({'value':{'p':55,'q':37},'derivation':'Calculated exactly.'})
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'answer_verification':True})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert not result.ready and 'fraction p/q' in result.issues[0]
    assert model.calls==critic.calls==1 and not critic.probe_cache
