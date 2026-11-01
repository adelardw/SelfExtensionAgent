"""Typed function results are data, with the same schema/routing/budget checks."""
import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel,ConfigDict,Field,ValidationError

from src.coding.model import OpenRouterModel,OutputBudgetExceeded


class Answer(BaseModel):
    model_config=ConfigDict(extra='forbid')
    count: int = Field(strict=True,ge=0)
    evidence: list[str] = Field(max_length=2)


def engine(monkeypatch,*,arguments=None,name='submit_result',calls=1,
           supported=('tools','reasoning'),mode='auto',stream=False,finish='tool_calls'):
    sent=[]
    arguments={'count':2,'evidence':['7']} if arguments is None else arguments
    def handle(request):
        if request.method=='GET':
            return httpx.Response(200,json={'data':[{'id':'small','supported_parameters':list(supported),
                'pricing':{'prompt':'0','completion':'0'}}]})
        payload=json.loads(request.content);sent.append(payload)
        tool_calls=[{'id':'result-'+str(i),'type':'function',
            'function':{'name':name,'arguments':json.dumps(arguments)}} for i in range(calls)]
        usage={'prompt_tokens':120,'completion_tokens':40,'cost':0}
        if stream:
            chunks=[{'choices':[{'delta':{'tool_calls':[dict(c,index=i) for i,c in enumerate(tool_calls)]}}]},
                    {'choices':[{'delta':{},'finish_reason':finish}],'usage':usage}]
            content=''.join('data: '+json.dumps(x)+'\n\n' for x in chunks)+'data: [DONE]\n\n'
            return httpx.Response(200,headers={'content-type':'text/event-stream'},content=content)
        message={'tool_calls':tool_calls,'content':None} if 'tools' in payload else {'content':json.dumps(arguments)}
        return httpx.Response(200,json={'choices':[{'finish_reason':finish,'message':message}],'usage':usage})
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handle),**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    model=OpenRouterModel({'model':'small','action_protocol':'tools','structured_output':mode,
        'cloud_context_tokens':32768,'max_output_tokens':256,'provider_only':['pinned'],
        'max_cost_usd':0,'max_input_usd_per_million':0,'max_output_usd_per_million':0})
    return model,sent


def invoke(model):
    return asyncio.run(model([{'role':'system','content':'Return the requested typed answer.'},
        {'role':'user','content':'{"task":"Count two records"}'}],Answer,'low'))


@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('supported',[['tools','reasoning'],['tools','response_format']])
def test_auto_uses_one_typed_function_when_json_schema_is_unavailable(monkeypatch,stream,supported):
    model,sent=engine(monkeypatch,stream=stream,supported=supported)
    assert invoke(model)==Answer(count=2,evidence=['7'])
    payload=sent[0]
    assert len(payload['tools'])==1
    tool=payload['tools'][0]['function']
    assert tool['name']=='submit_result' and tool['parameters']==Answer.model_json_schema()
    assert 'response_format' not in payload and 'tool_choice' not in payload
    assert payload['provider']['only']==['pinned'] and payload['provider']['allow_fallbacks'] is False
    assert payload['provider']['max_price']=={'prompt':0.,'completion':0.}
    assert 'OUTPUT JSON SCHEMA:' not in payload['messages'][0]['content']
    assert model.context_receipt['structured_tool_result'] is True
    assert model.calls==1 and model.usage()['tokens']==160 and model.provider_retries==0


@pytest.mark.parametrize('args',[
    {'count':'2','evidence':[]},{'count':2,'evidence':['1','2','3']},
    {'count':2,'evidence':[],'action':'run'},{'count':-1,'evidence':[]}])
def test_structured_call_arguments_still_require_exact_runtime_schema(monkeypatch,args):
    model,_=engine(monkeypatch,arguments=args)
    with pytest.raises(ValidationError):invoke(model)
    assert model.calls==1


@pytest.mark.parametrize('kwargs',[{'name':'read'},{'calls':0},{'calls':2}])
def test_arbitrary_or_ambiguous_function_calls_are_rejected(monkeypatch,kwargs):
    model,_=engine(monkeypatch,**kwargs)
    with pytest.raises(ValueError,match='submit_result'):invoke(model)


def test_truncated_typed_call_is_rejected_even_if_its_arguments_parse(monkeypatch):
    model,_=engine(monkeypatch,finish='length')
    with pytest.raises(OutputBudgetExceeded):invoke(model)


@pytest.mark.parametrize('mode,supported,expected',[
    ('text',['tools'],'text'),('json_object',['tools','response_format'],'json_object'),
    ('auto',['tools','structured_outputs'],'json_schema'),('auto',[],'text')])
def test_explicit_transport_and_provider_capabilities_are_preserved(monkeypatch,mode,supported,expected):
    model,sent=engine(monkeypatch,mode=mode,supported=supported,finish='stop')
    assert invoke(model).count==2
    assert 'tools' not in sent[0]
    assert model.context_receipt['response_transport']==expected
    assert 'structured_tool_result' not in model.context_receipt
