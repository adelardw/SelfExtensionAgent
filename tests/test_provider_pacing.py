"""Queued provider requests preserve cancellation, cost caps and per-call context."""
import asyncio
import json
import time

import httpx
import pytest

from src.coding.harness import Turn
from src.coding.model import OpenRouterModel,BudgetExceeded


def setup(monkeypatch,interval):
    starts=[]
    def handler(request):
        starts.append(time.monotonic())
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{
            'content':'{"action":"finish","summary":"inspection done"}'}}],
            'usage':{'prompt_tokens':100,'completion_tokens':20,'cost':.00001}})
    real=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    model=OpenRouterModel({'model':'small','max_output_tokens':64,'cloud_context_tokens':32768,
        'min_request_interval_seconds':interval})
    model.ready=True;model.rates=(.00000017,.00000025)
    return model,starts


def packet(task):
    return [{'role':'system','content':'Inspect only.'},
            {'role':'user','content':json.dumps({'task':task})}]


def test_parallel_callers_are_spaced_and_keep_their_own_context_receipts(monkeypatch):
    model,starts=setup(monkeypatch,.03)
    receipts=[]
    model.on_event=lambda kind,data:receipts.append(data) if kind=='request' else None
    async def run():
        return await asyncio.gather(*(model(packet('task '+'x'*length),Turn,'low') for length in [1,100,1000]))
    assert len(asyncio.run(run()))==3
    assert len(starts)==3 and all(b-a>=.025 for a,b in zip(starts,starts[1:]))
    assert model.calls==3 and model.reserved==0 and model.cost==pytest.approx(.00003)
    sizes=[event['receipt']['input_utf8_bytes'] for event in receipts]
    assert sizes[0]<sizes[1]<sizes[2]


def test_cancel_during_provider_wait_sends_nothing_and_reserves_no_spend(monkeypatch):
    model,starts=setup(monkeypatch,30)
    async def run():
        waiting=asyncio.Event()
        model.on_event=lambda kind,data:waiting.set() if kind=='provider-wait' else None
        await model(packet('first'),Turn,'low')
        pending=asyncio.create_task(model(packet('second'),Turn,'low'))
        await asyncio.wait_for(waiting.wait(),1)
        assert model.reserved==0
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):await pending
    asyncio.run(run())
    assert len(starts)==model.calls==1 and model.cost==.00001 and model.reserved==0


def test_queued_call_rechecks_spend_after_other_work_consumes_envelope(monkeypatch):
    model,starts=setup(monkeypatch,.02)
    model.config['max_cost_usd']=.01
    async def run():
        await model(packet('first'),Turn,'low')
        def event(kind,data):
            if kind=='provider-wait':model.cost=.01
        model.on_event=event
        with pytest.raises(BudgetExceeded):await model(packet('second'),Turn,'low')
    asyncio.run(run())
    assert len(starts)==model.calls==1 and model.reserved==0 and model.cost==.01
