"""Transient transport failures must not destroy useful agent work or multiply spend."""
import asyncio
import json

import httpx
import pytest

from src.coding.harness import Turn
from src.coding.model import OpenRouterModel,ProviderUnavailable,BudgetExceeded,OutputBudgetExceeded
from src.coding.provider_errors import rejection_details


PACKET=[{'role':'system','content':'Inspect only.'},{'role':'user','content':'{"task":"inspect"}'}]


def rejected(status=429,wait='1'):
    return httpx.Response(status,headers={'Retry-After':wait} if wait is not None else {},json={
        'error':{'message':'Provider returned error','metadata':{
            'raw':'Shared pool temporarily limited','provider_name':'Example',
            'provider_error_code':'model_rate_limit_exceeded','limit_source':'upstream_provider_shared_pool'}}})


def finished():
    return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{
        'content':'{"action":"finish","summary":"inspected"}'}}],
        'usage':{'prompt_tokens':20,'completion_tokens':10,'cost':0}})


def setup(monkeypatch,responses,**config):
    requests=[];waits=[]
    real_sleep=asyncio.sleep
    async def sleep(seconds):
        waits.append(seconds)
        await real_sleep(0)
    async def handle(request):
        requests.append(json.loads(request.content))
        await real_sleep(0)
        return responses.pop(0)
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handle),**kw))
    monkeypatch.setattr(asyncio,'sleep',sleep)
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    model=OpenRouterModel({'model':'small','max_output_tokens':64,'cloud_context_tokens':32768,
        'provider_only':['example/bf16'],'max_cost_usd':0,'max_input_usd_per_million':0,
        'max_output_usd_per_million':0,**config})
    model.ready=True
    return model,requests,waits


@pytest.mark.parametrize('bounded',[False,True])
def test_transient_rejection_recovers_same_request_with_shared_budget(monkeypatch,bounded):
    model,requests,waits=setup(monkeypatch,[rejected(),rejected(),finished()])
    call=model.bounded_call(PACKET,Turn,'low',32) if bounded else model(PACKET,Turn,'low')
    assert asyncio.run(call).action=='finish'
    assert len(requests)==3 and requests[0]==requests[1]==requests[2]
    assert requests[0]['max_tokens']==(32 if bounded else 64)
    assert requests[0]['provider']['only']==['example/bf16']
    assert requests[0]['provider']['allow_fallbacks'] is False
    assert waits==[1,2] and model.reserved==0 and model.cost==0
    assert model.calls==1 and model.failed_requests==2 and model.provider_error is None
    assert model.usage()['provider_retries']==2 and model.usage()['provider_retry_wait_seconds']==3


@pytest.mark.parametrize('status,wait',[(400,'1'),(401,'1'),(402,'1'),(429,None),(429,'nan'),
    (429,'infinity'),(429,'-1'),(429,'86400'),(503,None)])
def test_non_transient_or_unbounded_rejection_is_not_retried(monkeypatch,status,wait):
    model,requests,waits=setup(monkeypatch,[rejected(status,wait)])
    with pytest.raises(ProviderUnavailable):asyncio.run(model(PACKET,Turn,'low'))
    assert len(requests)==1 and not waits and model.provider_retries==0


def test_wait_budget_and_retry_count_are_global_across_roles(monkeypatch):
    model,requests,waits=setup(monkeypatch,[rejected(),finished(),rejected(),finished(),rejected()],
        provider_retry_wait_seconds=2)
    async def run():
        await model(PACKET,Turn,'low')
        await model.bounded_call(PACKET,Turn,'low',32)
        with pytest.raises(ProviderUnavailable):await model(PACKET,Turn,'low')
    asyncio.run(run())
    assert len(requests)==5 and waits==[1,1] and model.provider_retries==2


def test_parallel_callers_cannot_multiply_retry_allowance(monkeypatch):
    model,requests,waits=setup(monkeypatch,[rejected(),rejected(),rejected()],provider_retries=1)
    async def run():return await asyncio.gather(model(PACKET,Turn,'low'),model(PACKET,Turn,'low'),return_exceptions=True)
    results=asyncio.run(run())
    assert all(isinstance(r,ProviderUnavailable) for r in results)
    assert len(requests)==3 and waits==[1] and model.provider_retries==1 and model.reserved==0


def test_cancel_during_backoff_sends_no_extra_request(monkeypatch):
    model,requests,waits=setup(monkeypatch,[rejected(),finished()])
    async def run():
        pending=None
        def event(kind,data):
            if kind=='provider-retry':
                assert model.reserved==0
                pending.cancel()
        model.on_event=event
        pending=asyncio.create_task(model(PACKET,Turn,'low'))
        with pytest.raises(asyncio.CancelledError):await pending
    asyncio.run(run())
    assert len(requests)==1 and model.reserved==0 and model.calls==0


def test_spend_is_rechecked_after_backoff(monkeypatch):
    model,requests,waits=setup(monkeypatch,[rejected(),finished()],max_cost_usd=.01)
    def event(kind,data):
        if kind=='provider-retry':model.cost=.02
    model.on_event=event
    with pytest.raises(BudgetExceeded,match='spend envelope'):asyncio.run(model(PACKET,Turn,'low'))
    assert len(requests)==1 and model.reserved==0


def test_operator_steering_interrupts_backoff_before_a_stale_request(monkeypatch):
    from src.coding.control import SessionControl,Steered
    model,requests,waits=setup(monkeypatch,[rejected(),finished()])
    async def run():
        control=SessionControl()
        async def wait(seconds):await asyncio.Event().wait()
        monkeypatch.setattr(asyncio,'sleep',wait)
        # Mock transport also has its original zero-duration sleep, so it can run.
        def event(kind,data):
            if kind=='provider-retry':control.post('Use the revised task instead.')
        model.on_event=event
        with pytest.raises(Steered):
            await asyncio.wait_for(control.call(lambda:model(PACKET,Turn,'low')),1)
        assert control.drain()[0]['text']=='Use the revised task instead.'
    asyncio.run(run())
    assert len(requests)==1 and model.reserved==0


def test_output_exhaustion_never_uses_transport_retry(monkeypatch):
    response=httpx.Response(200,json={'choices':[{'finish_reason':'length','message':{'content':''}}],
        'usage':{'completion_tokens':64,'cost':0}})
    model,requests,waits=setup(monkeypatch,[response])
    with pytest.raises(OutputBudgetExceeded):asyncio.run(model(PACKET,Turn,'low'))
    assert len(requests)==1 and not waits and model.provider_retries==0


@pytest.mark.parametrize('body',[None,[],{'error':[]},{'error':{'metadata':[]}}])
def test_unusual_provider_errors_still_checkpoint(body):
    response=httpx.Response(429,content=json.dumps(body))
    assert rejection_details(response,'key')['message']=='Request rejected'


def test_diagnostic_retains_public_retry_reason_without_private_metadata():
    response=httpx.Response(429,headers={'Retry-After':'2'},json={'user_id':'private-user',
        'error':{'message':'key=secret','metadata':{'raw':'upstream rejected secret',
            'provider_name':'Test','provider_error_code':'rate_limit','limit_source':'shared',
            'headers':{'Authorization':'secret'},'request':{'private':'source'},'retry_after_seconds':5}}})
    details=rejection_details(response,'secret')
    assert details['retry_after_seconds']==2 and details['provider_error_code']=='rate_limit'
    assert '[redacted]' in details['message'] and 'upstream rejected' in details['message']
    assert 'secret' not in json.dumps(details) and 'private' not in json.dumps(details)
