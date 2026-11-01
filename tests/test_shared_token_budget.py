import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from src.coding.budget import TokenBudget, BudgetExceeded, token_scope, active_budget
from src.coding.model import OpenRouterModel, LocalModel, LlamaCppModel
from src.coding.decisions_api import DecisionsClient


class Answer(BaseModel):
    value: int


MESSAGES=[{'role':'system','content':'Return the answer.'},{'role':'user','content':'{}'}]


def cloud():
    m=OpenRouterModel({'model':'test','max_output_tokens':32,'provider_retries':0})
    m.ready=True
    return m


def transport(monkeypatch, handler):
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handler),**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')


def response(usage=None):
    data={'choices':[{'message':{'content':'{"value":1}'},'finish_reason':'stop'}]}
    if usage is not None:data['usage']=usage
    return httpx.Response(200,json=data)


def test_actual_usage_releases_unused_reservation_and_nested_scope_cannot_reset():
    with token_scope(100) as b:
        with b.reserve(90) as r:
            r.sent=True;r.usage({'prompt_tokens':10,'completion_tokens':20})
            with pytest.raises(BudgetExceeded):b.reserve(11)
        with token_scope(1000000) as nested:
            assert nested is b and b.spent==30 and b.reserved==0
            with pytest.raises(BudgetExceeded):b.reserve(71)
    assert active_budget.get() is None


def test_parallel_engines_and_critic_share_the_same_reservation(monkeypatch):
    async def scenario():
        started=asyncio.Event();release=asyncio.Event();posts=[]
        async def handler(request):
            posts.append(request)
            started.set();await release.wait()
            return response({'prompt_tokens':50,'completion_tokens':10,'cost':0})
        transport(monkeypatch,handler)
        with token_scope(1100) as budget:
            actor,critic=cloud(),cloud()
            pending=asyncio.create_task(actor(MESSAGES,Answer,'low'))
            await started.wait()
            reserved=budget.reserved
            assert 550<reserved<=1100
            with pytest.raises(BudgetExceeded):await critic(MESSAGES,Answer,'low')
            assert len(posts)==1 and budget.reserved==reserved
            release.set();await pending
            await critic(MESSAGES,Answer,'low')
            assert len(posts)==2 and budget.spent==120 and budget.reserved==0
    asyncio.run(scenario())


@pytest.mark.parametrize('failure',['missing_usage','malformed_usage','cancel'])
def test_unknown_or_cancelled_generation_keeps_conservative_charge(monkeypatch,failure):
    started=asyncio.Event()
    async def handler(request):
        started.set()
        if failure=='cancel':await asyncio.Event().wait()
        return response(None if failure=='missing_usage' else {'prompt_tokens':10})
    transport(monkeypatch,handler)
    async def scenario():
        with token_scope(1100) as budget:
            m=cloud();call=asyncio.create_task(m(MESSAGES,Answer,'low'))
            await started.wait()
            if failure=='cancel':
                call.cancel()
                with pytest.raises(asyncio.CancelledError):await call
            else:await call
            assert 550<budget.spent<=1100 and budget.estimated==budget.spent and budget.reserved==0
            with pytest.raises(BudgetExceeded):await cloud()(MESSAGES,Answer,'low')
    asyncio.run(scenario())


def test_rejected_request_releases_tokens_but_reported_overrun_is_never_hidden(monkeypatch):
    calls=[]
    def handler(request):
        calls.append(request)
        if len(calls)==1:return httpx.Response(429,json={'error':{'message':'rate limited'}})
        return response({'prompt_tokens':1500,'completion_tokens':2,'cost':0})
    transport(monkeypatch,handler)
    async def scenario():
        with token_scope(1100) as budget:
            with pytest.raises(BudgetExceeded):await cloud()(MESSAGES,Answer,'low')
            assert budget.spent==0 and budget.reserved==0
            await cloud()(MESSAGES,Answer,'low')
            assert budget.spent==1502 and budget.overruns==1
            with pytest.raises(BudgetExceeded):await cloud()(MESSAGES,Answer,'low')
            assert len(calls)==2
    asyncio.run(scenario())


def test_hydrated_prior_usage_counts_once_for_each_engine(monkeypatch):
    transport(monkeypatch,lambda r:response({'prompt_tokens':10,'completion_tokens':2,'cost':0}))
    async def scenario():
        m=cloud();m.input_tokens=700
        with token_scope(1100) as budget:
            for _ in range(2):
                with pytest.raises(BudgetExceeded):await m(MESSAGES,Answer,'low')
            assert budget.spent==700 and budget.reserved==0
    asyncio.run(scenario())


def test_output_cap_adapts_to_shared_remaining_capacity_without_increasing_it(monkeypatch):
    sent=[]
    def handler(request):
        sent.append(json.loads(request.content))
        return response({'prompt_tokens':100,'completion_tokens':10,'cost':0})
    transport(monkeypatch,handler)
    async def scenario():
        m=cloud();m.config['max_output_tokens']=8192
        with token_scope(2000) as budget:
            await m(MESSAGES,Answer,'low')
            capacity=m.context_receipt['output_capacity']
            assert capacity['requested']==8192
            assert 512<=capacity['admitted']<2000
            assert sent[0]['max_tokens']==capacity['admitted']
            assert budget.spent==110 and not budget.reserved
    asyncio.run(scenario())


def test_cancelled_harness_persists_unknown_usage_instead_of_losing_it(tmp_path,monkeypatch):
    from src.coding.harness import CodingHarness
    async def scenario():
        started=asyncio.Event()
        async def handler(request):
            started.set();await asyncio.Event().wait()
        transport(monkeypatch,handler)
        h=CodingHarness(tmp_path,model=cloud(),config={'learn':False})
        call=asyncio.create_task(h.run('Inspect the project'))
        await asyncio.wait_for(started.wait(),2)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):await call
        saved=json.loads(next(tmp_path.glob('.sea/coding/*/token-budget.json')).read_text())
        assert saved==h.token_budget.packet()==h.control.snapshot['token_budget']
        assert saved['accounted']==saved['estimated']>0 and saved['reserved']==0
    asyncio.run(scenario())


@pytest.mark.parametrize('kind',['cloud','ollama','llama_cpp','decisions'])
def test_every_provider_is_rejected_before_generation_when_shared_budget_is_empty(monkeypatch,kind):
    posts=[]
    def handler(request):
        posts.append(request)
        raise AssertionError('No generation may be sent')
    transport(monkeypatch,handler)
    async def scenario():
        with token_scope(1):
            if kind=='decisions':
                m=DecisionsClient({'model':'typesafe/test'});m.metadata={}
                call=lambda:m.request({}, {'q':{'type':'noul','instructions':'Relevant?'}})
            else:
                m={'cloud':cloud,'ollama':lambda:LocalModel({}),
                   'llama_cpp':lambda:LlamaCppModel({})}[kind]()
                m.ready=True
                call=lambda:m(MESSAGES,Answer,'low')
            with pytest.raises(BudgetExceeded):await call()
            assert not posts and m.calls==0
    asyncio.run(scenario())
