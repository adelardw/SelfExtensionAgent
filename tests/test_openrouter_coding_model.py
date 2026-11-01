"""Cheap-only cloud profile, streamed cancellation accounting and spend reservations."""
import asyncio
import json

import httpx
import pytest

from src.coding.harness import Turn, CodingHarness
from src.coding.model import OpenRouterModel, BudgetExceeded, OutputBudgetExceeded, build_local_model


def completion_transport(monkeypatch,choice,usage,*,streaming=False):
    requests=[]
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            delta=choice.get('message',{})
            for chunk in [{'choices':[{'delta':delta,'finish_reason':choice.get('finish_reason')}]},
                          {'choices':[],'usage':usage}]:
                yield ('data: '+json.dumps(chunk)+'\n\n').encode()
            yield b'data: [DONE]\n\n'
    def handler(request):
        requests.append(json.loads(request.content))
        if streaming:return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Stream())
        return httpx.Response(200,json={'choices':[choice],'usage':usage})
    real=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    return requests


@pytest.mark.parametrize('sampling,expected',[
    ({},{}),({'temperature':None,'top_p':None},{}),
    ({'temperature':0},{'temperature':0}),
    ({'temperature':1.0,'top_p':.95},{'temperature':1.0,'top_p':.95}),
])
def test_cloud_sampling_uses_explicit_controls_or_provider_defaults(monkeypatch,sampling,expected):
    requests=completion_transport(monkeypatch,
        {'message':{'content':'{"action":"finish","summary":"ok"}'},'finish_reason':'stop'},
        {'prompt_tokens':100,'completion_tokens':20,'cost':0})
    model=OpenRouterModel({'model':'cheap',**sampling});model.ready=True
    asyncio.run(model([{'role':'system','content':'policy'},{'role':'user','content':'{}'}],Turn,'low'))
    assert {k:requests[0][k] for k in ('temperature','top_p') if k in requests[0]}==expected
    assert model.context_receipt['sampling']==expected


@pytest.mark.parametrize('sampling',[
    {'temperature':float('nan')},{'temperature':float('inf')},{'temperature':-1},
    {'temperature':3},{'temperature':True},{'top_p':1.1},{'top_p':-0.1},{'top_p':'oops'},
])
def test_invalid_cloud_sampling_fails_before_any_request(sampling):
    with pytest.raises(ValueError,match='sampling'):
        OpenRouterModel({'model':'cheap',**sampling})


@pytest.mark.parametrize('streaming',[False,True])
@pytest.mark.parametrize('finish_reason',['length',None])
def test_exhausted_reasoning_is_accounted_as_budget_failure_not_json_repair(monkeypatch,streaming,finish_reason):
    completion_transport(monkeypatch,{'message':{'content':''},'finish_reason':finish_reason},
        {'prompt_tokens':100,'completion_tokens':32,'completion_tokens_details':{'reasoning_tokens':32},'cost':.00005},
        streaming=streaming)
    model=OpenRouterModel({'model':'cheap','max_output_tokens':32})
    model.ready=True;model.rates=(.00000017,.00000025)
    with pytest.raises(OutputBudgetExceeded) as captured:
        asyncio.run(model([{'role':'system','content':'policy'},{'role':'user','content':'{}'}],Turn,'medium'))
    assert captured.value.details['visible_chars']==0
    usage=model.usage()
    assert usage['calls']==1 and usage['truncated_outputs']==1 and usage['reasoning_tokens']==32
    assert usage['reasoning_usage_calls']==1 and usage['last_output']['max_output_tokens']==32
    assert usage['cost']==.00005 and not usage['cost_estimated'] and model.reserved==0


@pytest.mark.parametrize('contract',[False,True])
def test_output_limit_checkpoints_after_one_request_without_retries(tmp_path,monkeypatch,contract):
    requests=completion_transport(monkeypatch,{'message':{'content':''},'finish_reason':'length'},
        {'prompt_tokens':100,'completion_tokens':32,'cost':.00005})
    model=OpenRouterModel({'model':'cheap','max_output_tokens':32})
    model.ready=True;model.rates=(.00000017,.00000025)
    config={'orchestration':{'enabled':True,'auto_domains':True,'same_model':True,'critic':{'enabled':True}}} if contract else {}
    result=asyncio.run(CodingHarness(tmp_path,model=model,config=config).run('Inspect the project'))
    assert result['status']=='paused' and len(requests)==1
    assert 'output budget exhausted' in result['final_answer']
    assert __import__('pathlib').Path(result['checkpoint']).is_file()


def test_valid_native_arguments_at_length_limit_never_execute(tmp_path,monkeypatch):
    requests=completion_transport(monkeypatch,{'finish_reason':'length','message':{'content':'','tool_calls':[
        {'id':'call1','type':'function','function':{'name':'write','arguments':json.dumps({'path':'must_not_exist','content':'truncated proposal'})}}]}},
        {'prompt_tokens':100,'completion_tokens':32,'cost':.00005})
    model=OpenRouterModel({'model':'cheap','max_output_tokens':32,'cloud_context_tokens':32768,'action_protocol':'tools'})
    model.ready=True;model.rates=(.00000017,.00000025)
    result=asyncio.run(CodingHarness(tmp_path,model=model,config={}).run('Write a file'))
    assert result['status']=='paused' and len(requests)==1 and not (tmp_path/'must_not_exist').exists()


def test_malformed_under_limit_remains_an_ordinary_validation_error(monkeypatch):
    from pydantic import ValidationError
    completion_transport(monkeypatch,{'message':{'content':'not json'},'finish_reason':'stop'},
        {'prompt_tokens':100,'completion_tokens':5,'cost':.00005})
    model=OpenRouterModel({'model':'cheap','max_output_tokens':32})
    model.ready=True;model.rates=(.00000017,.00000025)
    with pytest.raises(ValidationError):
        asyncio.run(model([{'role':'system','content':'policy'},{'role':'user','content':'{}'}],Turn,'low'))
    assert model.truncated_outputs==0 and model.usage()['last_output']['reasoning_tokens'] is None


@pytest.mark.parametrize('meta,effort,expected',[
    ({},'medium',{'enabled':True}),
    ({'supported_efforts':None},'medium',{'effort':'medium'}),
    ({'supported_efforts':['low','high']},'medium',{'effort':'low'}),
    ({'mandatory':True},'low',{'enabled':True}),
    ({},'low',{'enabled':False}),
])
def test_reasoning_payload_respects_advertised_controls(monkeypatch,meta,effort,expected):
    requests=completion_transport(monkeypatch,{'message':{'content':'{"action":"finish","summary":"ok"}'},'finish_reason':'stop'},
        {'prompt_tokens':100,'completion_tokens':20,'cost':.00005})
    model=OpenRouterModel({'model':'cheap'});model.ready=True;model.rates=(.00000017,.00000025)
    model.reasoning_meta=meta
    asyncio.run(model([{'role':'system','content':'policy'},{'role':'user','content':'{}'}],Turn,effort))
    assert requests[0]['reasoning']==expected


@pytest.mark.parametrize('config,meta,expected',[
    ({'max_output_tokens':3500},{},3500),
    ({},{},1536),
    ({},{'mandatory':True},8192),
])
def test_output_capacity_does_not_silently_shrink_with_low_effort(monkeypatch,config,meta,expected):
    requests=completion_transport(monkeypatch,{'message':{'content':'{"action":"finish","summary":"ok"}'},'finish_reason':'stop'},
        {'prompt_tokens':100,'completion_tokens':20,'cost':.00005})
    model=OpenRouterModel({'model':'cheap','cloud_context_tokens':32768,**config})
    model.ready=True;model.rates=(.00000017,.00000025);model.reasoning_meta=meta
    asyncio.run(model([{'role':'system','content':'policy'},{'role':'user','content':'{}'}],Turn,'low'))
    assert requests[0]['max_tokens']==expected


def test_free_only_rejects_paid_catalog_before_generation(monkeypatch):
    requests=[]
    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200,json={'data':[{'id':'paid','pricing':{'prompt':'.00000001','completion':'0'}}]})
    real=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    model=OpenRouterModel({'model':'paid','max_cost_usd':0,'max_input_usd_per_million':0,'max_output_usd_per_million':0})
    with pytest.raises(ValueError,match='price ceiling'):
        asyncio.run(model([{'role':'system','content':'policy'},{'role':'user','content':'{}'}],Turn,'low'))
    assert requests==['/api/v1/models']


def test_gemini_wire_schema_preserves_tools_and_runtime_validation():
    from src.coding.request import cloud_schema
    from src.coding.critic import Assessment
    from src.coding.profiles import ProfileSpec
    from pydantic import ValidationError
    wire=cloud_schema(Assessment,'google/gemini-3.1-flash-lite')
    tools=wire['$defs']['ProfileSpec']['properties']['tools']
    assert 'maxItems' not in tools and 'edit' in tools['items']['enum']
    assert cloud_schema(Assessment,'other/model')['$defs']['ProfileSpec']['properties']['tools']['maxItems']==32
    assert Assessment.model_json_schema()['properties']['criteria']['maxItems']==24
    with pytest.raises(ValidationError):
        ProfileSpec(name='bad',base='reason',strategy='A valid strategy',tools=['read']*33,exit_when=['done'])
    with pytest.raises(ValidationError):
        ProfileSpec(name='bad',base='reason',strategy='A valid strategy',tools=['unknown-tool'],exit_when=['done'])


def test_cloud_is_only_used_when_explicitly_selected():
    assert isinstance(build_local_model({'backend':'openrouter'}),OpenRouterModel)
    with pytest.raises(ValueError): build_local_model({'backend':'auto'})


def test_compact_packet_preserves_constraints_and_deduplicates_only_exact_source():
    from src.coding.request import compact_packet,schema_hint
    from src.coding.critic import TaskContract
    packet={'task':'review','human_steering':[{'id':'u1','text':'Never edit'}],
            'inspected_content':{'a.py':'exact source'},
            'recent':[{'tool':'read','output':'exact source','evidence_id':'12'},
                      {'tool':'read','output':'different range','evidence_id':'13'}]}
    compact=compact_packet(packet)
    assert compact['human_steering']==packet['human_steering']
    assert compact['recent'][0]['evidence_id']=='12' and 'inspected_content' in compact['recent'][0]['output']
    assert compact['recent'][1]['output']=='different range' and packet['recent'][0]['output']=='exact source'
    hint=schema_hint(TaskContract)
    assert 'response_format' in hint and '"text","json"' in hint and 'domains' in hint


def test_price_ceiling_rejects_expensive_model_before_completion(monkeypatch):
    seen=[]
    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200,json={'data':[{'id':'expensive','pricing':{'prompt':'.000002','completion':'.00002'}}]})
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:real(transport=transport,**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    model=OpenRouterModel({'model':'expensive'})
    with pytest.raises(ValueError,match='price ceiling'):
        asyncio.run(model([{'role':'system','content':'s'},{'role':'user','content':'{}'}],Turn,'low'))
    assert seen == ['/api/v1/models']


def test_cloud_streaming_receipt_usage_reasoning_and_no_fallback(monkeypatch):
    events=[]; requests=[]
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            chunks=[{'choices':[{'delta':{'reasoning':'provider summary'}}]},
                    {'choices':[{'delta':{'content':'{"action":"finish","summary":"ok"}'}}]},
                    {'choices':[],'usage':{'prompt_tokens':100,'completion_tokens':20,'cost':.000055}}]
            for chunk in chunks: yield ('data: '+json.dumps(chunk)+'\n\n').encode()
            yield b'data: [DONE]\n\n'
    def handler(request):
        requests.append(request)
        if request.url.path.endswith('/models'):
            return httpx.Response(200,json={'data':[{'id':'cheap','pricing':{'prompt':'.00000025','completion':'.0000015'}}]})
        payload=json.loads(request.content)
        assert payload['model']=='cheap' and 'models' not in payload
        assert payload['provider']['max_price']['completion']==2
        assert payload['provider']['only']==['test-provider/bf16'] and payload['provider']['allow_fallbacks'] is False
        assert payload['response_format']['json_schema']['strict'] is True
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Stream())
    real=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    model=OpenRouterModel({'model':'cheap','provider_only':['test-provider/bf16']})
    model.on_event=lambda kind,data:events.append(kind)
    result=asyncio.run(model([{'role':'system','content':'policy'},{'role':'user','content':'{"task":"inspect"}'}],Turn,'low'))
    assert result.summary=='ok' and model.usage()['cost']==.000055
    assert not model.usage()['cost_estimated'] and model.reserved==0
    assert events==['request','reasoning','content','response','validated','end']
    assert len(requests)==2


def test_spend_budget_pauses_without_model_request(tmp_path):
    class BudgetModel:
        async def __call__(self,*args): raise BudgetExceeded('Envelope reached')
    result=asyncio.run(CodingHarness(tmp_path,model=BudgetModel(),config={}).run('inspect'))
    assert result['status']=='paused' and result['final_answer']=='Envelope reached'


@pytest.mark.parametrize('status',[400,402,429,503])
def test_provider_rejection_preserves_checkpoint_diagnostic_and_does_not_retry(tmp_path,monkeypatch,status):
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(status,json={'error':{'message':'Provider returned error',
            'metadata':{'raw':json.dumps({'error':{'message':'Rejected test-key request'}})}}})
    real=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    model=OpenRouterModel({'model':'cheap'})
    model.ready=True;model.rates=(.00000025,.0000015)
    result=asyncio.run(CodingHarness(tmp_path,model=model,config={}).run('Inspect the project'))
    assert result['status']=='paused' and len(requests)==1
    assert str(status) in result['final_answer'] and 'Rejected [redacted] request' in result['final_answer']
    assert 'test-key' not in json.dumps(result)
    assert result['failed_requests']==1 and result['provider_error']['status']==status
    assert __import__('pathlib').Path(result['checkpoint']).is_file()


def test_shared_reservation_blocks_concurrent_overspend(monkeypatch):
    async def scenario():
        started,finish=asyncio.Event(),asyncio.Event()
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set();await finish.wait()
                yield b'data: {"choices":[{"delta":{"content":"{\\"action\\":\\"finish\\",\\"summary\\":\\"ok\\"}"}}]}\n\n'
                yield b'data: [DONE]\n\n'
        def handler(request): return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Stream())
        real=httpx.AsyncClient
        monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
        import src.llm.llm as llm
        monkeypatch.setattr(llm,'api_key',lambda:'test-key')
        model=OpenRouterModel({'max_cost_usd':.005})
        model.ready=True;model.rates=(.00000025,.0000015)
        messages=[{'role':'system','content':'policy'},{'role':'user','content':'{}'}]
        first=asyncio.create_task(model(messages,Turn,'low'))
        await started.wait()
        # One reservation fits; two would exceed the same run envelope.
        with pytest.raises(BudgetExceeded): await model(messages,Turn,'low')
        finish.set();await first
        assert model.reserved==0 and model.estimated_cost
    asyncio.run(scenario())
