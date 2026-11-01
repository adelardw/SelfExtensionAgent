"""Provider grammar capabilities do not change the action permission boundary."""
import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from src.coding.harness import Turn
from src.coding.model import OpenRouterModel
from src.coding.orchestration import scoped_schema
from src.coding.request import response_transport,normalize_json_envelope


@pytest.mark.parametrize('supported,expected',[
    (None,'json_schema'),(['structured_outputs','response_format'],'json_schema'),
    (['response_format'],'json_object'),(['tools','reasoning'],'text'),([], 'text'),
])
def test_auto_uses_declared_capabilities(supported,expected):
    assert response_transport({},supported)==expected


def test_explicit_selection_is_not_silently_changed():
    assert response_transport({'structured_output':'json_schema'},[])=='json_schema'
    with pytest.raises(ValueError):response_transport({'structured_output':'unknown'},[])


@pytest.mark.parametrize('action',[
    {'action':'plan','plan':['repeat']},
    {'action':'delegate','role':'inspect','task':'repeat'},
    {'action':'finish','summary':'not verified'},
])
def test_blocked_action_validation_does_not_depend_on_provider_decoder(action):
    schema=scoped_schema(Turn,{'read'},can_delegate=False,can_plan=False,blocked_actions={'finish'})
    with pytest.raises(ValidationError):schema.model_validate(action)


@pytest.mark.parametrize('mode',['json_schema','json_object','text'])
@pytest.mark.parametrize('valid',[True,False])
def test_every_transport_validates_scoped_actions_and_preserves_routing(monkeypatch,mode,valid):
    supported={'json_schema':['structured_outputs'],'json_object':['response_format'],'text':[]}[mode]
    requests=[]
    schema=scoped_schema(Turn,{'read'},can_delegate=False,can_plan=False)
    def handle(request):
        if request.method=='GET':
            return httpx.Response(200,json={'data':[{'id':'small','pricing':{'prompt':'0','completion':'0'},
                'supported_parameters':supported}]})
        payload=json.loads(request.content);requests.append(payload)
        action={'action':'tool','tool':'read' if valid else 'write','args':{'path':'file.py'}}
        content=json.dumps(action)+('}' if mode in {'json_object','text'} else '')
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':content}}],
            'usage':{'prompt_tokens':120,'completion_tokens':30,'cost':0}})
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handle),**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    model=OpenRouterModel({'model':'small','max_output_tokens':64,'cloud_context_tokens':32768,
        'provider_only':['only-provider'],'max_cost_usd':0,'max_input_usd_per_million':0,'max_output_usd_per_million':0})
    packet=[{'role':'system','content':'Use permitted actions only.'},{'role':'user','content':'{"task":"inspect"}'}]
    call=model(packet,schema,'low')
    if valid:assert asyncio.run(call).tool=='read'
    else:
        with pytest.raises(ValidationError):asyncio.run(call)
    assert len(requests)==1 and model.calls==1 and model.provider_retries==0
    sent=requests[0]
    assert sent['model']=='small' and sent['provider']['only']==['only-provider']
    assert sent['provider']['allow_fallbacks'] is False
    assert model.context_receipt['response_transport']==mode
    assert model.json_envelope_repairs==int(mode in {'json_object','text'})
    assert packet[0]['content']=='Use permitted actions only.'
    if mode=='json_schema':
        assert sent['response_format']['type']=='json_schema'
        assert 'OUTPUT JSON SCHEMA' not in sent['messages'][0]['content']
    else:
        assert ('response_format' in sent)==(mode=='json_object')
        if mode=='json_object':assert sent['response_format']=={'type':'json_object'}
        assert sent['messages'][0]['content'].count('OUTPUT JSON SCHEMA:')==1
        assert json.dumps(schema.model_json_schema(),ensure_ascii=False,separators=(',',':')) in sent['messages'][0]['content']


def test_redundant_closer_preserves_exact_values_and_still_checks_permissions():
    schema=scoped_schema(Turn,{'read'},can_delegate=False)
    raw='{"action":"tool","tool":"read","args":{"path":"file[1].py"},"summary":"a } inside a string"}'
    normalized,repair=normalize_json_envelope(raw+'}}')
    assert normalized==raw and repair=='surplus_closing_delimiters'
    assert schema.model_validate_json(normalized).args['path']=='file[1].py'
    forbidden=raw.replace('"tool":"read"','"tool":"write"')+'}'
    with pytest.raises(ValidationError):schema.model_validate_json(normalize_json_envelope(forbidden)[0])


@pytest.mark.parametrize('text',[
    '{"action":"finish","summary":"partial"',
    '{"action":"finish","summary":"first"}{"action":"finish","summary":"second"}',
    'prefix {"action":"finish","summary":"ambiguous"}',
    '{"action":"finish","summary":"ambiguous"} trailing instructions',
    '{"action":"finish","summary":"ok"}',
    '[{"action":"finish","summary":"ok"}]]',
    '{"action":"finish","summary":"ok"}'+('}'*9),
])
def test_envelope_repair_does_not_invent_or_choose_actions(text):
    assert normalize_json_envelope(text)==(text,None)
