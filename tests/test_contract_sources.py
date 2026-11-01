"""Exact source references replace fragile quotation generation across languages."""
import json

import pytest
from pydantic import ValidationError

from src.coding.contract_sources import source_packet,source_contract
from src.coding.critic import TaskContract
from src.coding.request import fit_messages


@pytest.mark.parametrize('task',[
    'Add tests and run checks. Return JSON with status and tests.',
    'Вычисли точно.\nНе создавай файлы; верни результат.',
    'Preserve code:\n    def parse(value):\n        return "x.y; z"\n',
    ('long identifier'*40)+'\n'+('условие '*160),
])
def test_all_input_characters_survive_ordered_bounded_blocks(task):
    steering=[{'id':'u1','text':'Keep the API.\nНе публикуй.','target':'main','origin':'user'}]
    packet,index=source_packet(task,steering)
    assert ''.join(b['text'] for b in packet['task'])==task
    assert ''.join(b['text'] for b in packet['human_steering'][0]['text'])==steering[0]['text']
    assert packet['human_steering'][0]['id']=='u1' and steering[0]['text']=='Keep the API.\nНе публикуй.'
    assert all(len(s)<=400 for s in index.values())
    assert len(index)==len(packet['task'])+len(packet['human_steering'][0]['text'])


def test_source_id_binds_to_original_input_and_wire_does_not_require_copying_quote():
    task='Add regression tests and run the registered unit check.'
    packet,index=source_packet(task,[])
    schema=source_contract(TaskContract,index)
    props=schema.model_json_schema()['$defs']['Requirement']
    assert 'source_id' in props['required'] and 'source_quote' not in props['properties']
    result=schema.model_validate({'requirements':[
        {'id':'r1','text':'Add regression tests','validation':'new_or_updated_tests','source_id':'t0'},
        {'id':'r2','text':'Run checks','validation':'registered_checks','source_id':'t0'}]},context={'sources':[task]})
    assert {r.source_quote for r in result.requirements}=={task}
    assert 'source_id' not in result.model_dump()['requirements'][0]
    assert schema.__name__=='TaskContract'


@pytest.mark.parametrize('extra',[
    {'source_id':'invented'}, {'source_id':1}, {'source_id':'t0','source_quote':'different instruction'},
])
def test_missing_or_conflicting_source_references_cannot_authorize_work(extra):
    _,index=source_packet('Add tests.',[])
    schema=source_contract(TaskContract,index)
    with pytest.raises(ValidationError):
        schema.model_validate({'requirements':[{'id':'r1','text':'Add tests',
            'validation':'new_or_updated_tests',**extra}]})


def test_binding_a_valid_pointer_does_not_make_its_semantics_authoritative():
    task='Return JSON with value and derivation.'
    _,index=source_packet(task,[])
    schema=source_contract(TaskContract,index)
    with pytest.raises(ValidationError,match='JSON answer format'):
        schema.model_validate({'requirements':[{'id':'r1','text':'Return JSON',
            'validation':'new_or_updated_tests','source_id':'t0'}]},context={'sources':[task]})


def test_context_compaction_cannot_silently_clip_referenced_operator_blocks():
    packet,index=source_packet('Inspect.',[{'id':'u1','text':'constraint '*5000}])
    with pytest.raises(ValueError,match='steering'):
        fit_messages([{'role':'system','content':'extract requirements'},
            {'role':'user','content':json.dumps(packet)}],source_contract(TaskContract,index),16384,2048)
