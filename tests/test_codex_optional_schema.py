"""Regression from the real ReviewSchema HTTP 400 in campaign 600."""
import copy
import json
from pydantic import BaseModel, ConfigDict
from scripts.codex_behavior_eval import codex_envelope, decode_codex_payload


class Profile(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str
    notes:list[str]=[]


class Review(BaseModel):
    model_config=ConfigDict(extra='forbid')
    ready:bool
    profile:Profile|None=None
    next_steps:list[str]=[]


def test_optional_critic_fields_have_a_provider_compatible_wire_schema():
    wire=Review.model_json_schema();original=copy.deepcopy(wire)
    envelope,encoding=codex_envelope(wire,True)
    assert encoding=='native_object'
    def verify(node):
        if isinstance(node,list):
            for item in node:verify(item)
        elif isinstance(node,dict):
            if node.get('type')=='object':
                assert set(node['required'])==set(node['properties'])
                assert node['additionalProperties'] is False
            for item in node.values():verify(item)
    verify(envelope)
    assert wire==original,'Transport normalization must not mutate the runtime schema'
    decoded=decode_codex_payload({'ready':False,'profile':None,'next_steps':[]},encoding)
    assert Review.model_validate_json(decoded)==Review(ready=False)
    decoded=decode_codex_payload({'ready':False,'profile':{'name':'repair','notes':[]},'next_steps':['Build UI']},encoding)
    assert Review.model_validate_json(decoded).profile.name=='repair'
