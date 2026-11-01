import asyncio
import json

import httpx
import pytest

from src.coding.decision import DecisionRouter
from src.coding.decisions_api import DecisionsClient


def state(**updates):
    return {'phase':'implement','allowed_tools':['read','edit','write','check','web'],
            'failures':2,'profiles':[],'human_steering':[{'id':'u1','text':'Preserve the API'}],
            'budget':{'nodes_left':3,'depth_left':2},'trigger':'observation',**updates}


class Decider:
    name='typesafe/jev-1.13'
    calls=0
    def __init__(self,choice,score=.95): self.choice,self.score,self.requests=choice,score,[]
    async def request(self,packet,questions):
        self.calls+=1;self.requests.append((packet,questions))
        options=questions['route']['criteria']
        scores={k:(self.score if k==self.choice else (1-self.score)/(len(options)-1)) for k in options}
        return {'route':{'choice':self.choice,'probabilities':scores}}
    def usage(self): return {'calls':self.calls,'cost':self.calls*.00001,'tokens':self.calls*100}


class Planner:
    calls=0
    async def __call__(self,messages,schema,effort):
        self.calls+=1
        return schema.model_validate({'operation':'fork','mode':'plan','branches':[
            {'mode':'inspect','task':'Inspect parser','tools':['read']},
            {'mode':'review','task':'Review unrelated export API','tools':['read']}],
            'reason':'Independent scopes','rules':[]})
    def usage(self): return {'calls':self.calls,'cost':self.calls*.001,'tokens':self.calls*200}


def router(choice,score=.95,**config):
    result=DecisionRouter(object(),{'backend':'decisions','max_calls':4,**config})
    result.remote=Decider(choice,score);result.planner=Planner()
    return result


def test_existing_mode_needs_no_generation_and_preserves_operator_constraints():
    r=router('mode_repair')
    result=asyncio.run(r.choose('Fix reproduced parser bug',state()))
    assert result.operation=='switch' and result.mode=='repair' and result.rules==['stop-repetition']
    assert r.planner.calls==0 and r.usage()['calls']==1
    assert r.remote.requests[0][0]['human_steering'][0]['text']=='Preserve the API'
    assert asyncio.run(r.choose('Fix reproduced parser bug',state())) is result
    assert r.remote.calls==1


def test_uncertain_choice_keeps_current_mode_without_paid_escalation():
    r=router('mode_review',.2)
    result=asyncio.run(r.choose('Continue',state()))
    assert result.operation=='continue' and result.mode=='implement'
    assert r.planner.calls==0 and 'Uncertain' in result.reason


def test_plan_and_readonly_ceiling_remove_writing_modes_before_request():
    r=router('mode_plan')
    asyncio.run(r.choose('Plan only',state(execution='plan')))
    options=r.remote.requests[-1][1]['route']['criteria']
    assert 'mode_implement' not in options and 'mode_repair' not in options
    r=router('mode_review')
    asyncio.run(r.choose('Inspect',state(allowed_tools=['read'])))
    options=r.remote.requests[-1][1]['route']['criteria']
    assert not {'mode_implement','mode_repair','mode_research','mode_verify'} & options.keys()


def test_composition_is_explicit_bounded_and_both_model_costs_are_counted():
    r=router('compose',planner={'max_calls':1})
    result=asyncio.run(r.choose('Two independent modules',state(trigger='delegation')))
    assert result.operation=='fork' and len(result.branches)==2
    assert r.planner.calls==1 and r.cost==pytest.approx(.00101) and r.tokens==300
    r.remote.choice='mode_review'
    asyncio.run(r.choose('Next task',state()))
    assert 'compose' not in r.remote.requests[-1][1]['route']['criteria']


def test_delegation_selects_a_worker_profile_without_inventing_subtasks():
    r=router('mode_review')
    result=asyncio.run(r.choose('Review patch',state(trigger='delegation')))
    assert result.operation=='select' and result.mode=='review' and not result.branches


@pytest.mark.parametrize('failure',['unknown_choice','foreign_probabilities','nan','sum','not_max','confidence','answer_type','scores_type','choice_type'])
def test_decisions_transport_rejects_malformed_choice_but_accounts_for_it(monkeypatch,failure):
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    answer={'choice':'a','probabilities':{'a':.9,'b':.1},'confidence':.8}
    if failure=='unknown_choice': answer['choice']='foreign'
    elif failure=='foreign_probabilities': answer['probabilities']={'a':.9,'c':.1}
    elif failure=='nan': answer['probabilities']['a']='nan'
    elif failure=='sum': answer['probabilities']={'a':.9,'b':.9}
    elif failure=='not_max': answer['choice']='b'
    elif failure=='answer_type': answer=[]
    elif failure=='scores_type': answer['probabilities']=[]
    elif failure=='choice_type': answer['choice']=[]
    else: answer['confidence']=2
    def handle(request):
        if request.method=='GET':
            return httpx.Response(200,json={'data':{'id':'typesafe/jev-1.13',
                'architecture':{'output_modalities':['decisions']},
                'endpoints':[{'pricing':{'prompt':'0.000000042','completion':'0'}}]}})
        return httpx.Response(200,json={'answers':{'route':answer},'usage':{'cost':.00001}})
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handle),**kw))
    client=DecisionsClient({})
    with pytest.raises(ValueError):
        asyncio.run(client.request({'task':'review'},{'route':{'type':'choice','instructions':'Choose','criteria':{'a':'A','b':'B'}}}))
    assert client.calls==1 and client.cost==pytest.approx(.00001)


def test_decision_budget_rejects_large_context_without_truncating_human_steering(monkeypatch):
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    client=DecisionsClient({'max_input_bytes':200})
    with pytest.raises(ValueError,match='constraint truncation'):
        asyncio.run(client.request({'human_steering':'important '*100},
            {'route':{'type':'choice','instructions':'Choose','criteria':{'a':'A','b':'B'}}}))
    assert client.calls==0


@pytest.mark.parametrize('mode,operation',[('direct','select'),('read','select'),('review','create')])
def test_generative_planner_rejects_unknown_modes_and_incomplete_creation(mode,operation):
    async def planner(messages,schema,effort):
        assert 'available_modes' in messages[-1]['content']
        return {'operation':operation,'mode':mode,'branches':[],'rules':[],'reason':'routing'}
    r=DecisionRouter(planner,{'backend':'local'})
    with pytest.raises(ValueError): asyncio.run(r.choose('Review a change',state()))


def test_generative_profile_name_is_used_without_renaming_an_existing_mode():
    async def planner(messages,schema,effort):
        return {'operation':'create','mode':'reason','branches':[],'rules':[],'reason':'specialized review',
                'profile':{'name':'invariant-review','base':'reason','strategy':'Check the loop invariant against every exit.',
                           'tools':['read'],'exit_when':['Every exit inspected']}}
    r=DecisionRouter(planner,{'backend':'local'})
    result=asyncio.run(r.choose('Review invariant',state()))
    assert result.mode=='invariant-review' and result.profile.base=='reason'


def test_concurrent_routing_uses_one_budget_and_cache():
    async def run():
        r=router('mode_review',max_calls=1)
        outcomes=await asyncio.gather(*(r.choose(task,state()) for task in ('same','same','different')))
        assert outcomes[0] is outcomes[1] and outcomes[2] is None
        assert r.remote.calls==1 and r.calls==1
    asyncio.run(run())
