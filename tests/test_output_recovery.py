"""Reasoning-to-action recovery is opt-in, bounded, accounted, and scope preserving."""
import asyncio
import copy
import json
import sys

import httpx
import pytest
from pydantic import BaseModel

from src.coding.harness import CodingHarness,Turn
from src.coding.model import OpenRouterModel,OutputBudgetExceeded,BudgetExceeded
from src.coding.output_recovery import ProviderDraft
from src.runtime import hitl


def truncated(draft='An incomplete provider draft; inspect the current source before deciding.'):
    return {'message':{'content':'','reasoning':draft},'finish_reason':'length'}


def finished(value):
    return {'message':{'content':json.dumps(value)},'finish_reason':'stop'}


def setup(monkeypatch,replies,**config):
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        choice=replies[len(requests)-1]
        return httpx.Response(200,json={'choices':[choice],'provider':'Fixed',
            'usage':{'prompt_tokens':100,'completion_tokens':64 if choice['finish_reason']=='length' else 20,
                     'cost':config.get('mock_cost',.00001)}})
    real=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    model=OpenRouterModel({'model':'cheap','provider_only':['fixed'],'max_output_tokens':64,
        'cloud_context_tokens':32768,'output_recovery':True,**config})
    model.ready=True;model.rates=(.00000017,.00000025)
    return model,requests


def messages():
    return [{'role':'system','content':'Never edit files without task authority.'},
            {'role':'user','content':json.dumps({'task':'Inspect only','human_steering':[{'id':'u1','text':'Do not write files'}],
                'checks':{'unit':['python3','-m','unittest']},'obligations':{'requirements':[{'id':'r1','text':'Inspection only'}]}})}]


def test_same_model_serializes_returned_draft_once_and_accounts_both_requests(monkeypatch):
    model,requests=setup(monkeypatch,[truncated(),finished({'action':'tool','tool':'read','args':{'path':'module.py'}})],
        output_recovery_tokens=32)
    original=messages();before=copy.deepcopy(original)
    result=asyncio.run(model(original,Turn,'medium'))
    assert result.tool=='read' and original==before
    assert len(requests)==2 and requests[0]['model']==requests[1]['model']=='cheap'
    assert requests[0]['provider']==requests[1]['provider'] and requests[1]['provider']['allow_fallbacks'] is False
    assert requests[0]['reasoning']=={'enabled':True} and requests[1]['reasoning']=={'enabled':False}
    assert requests[1]['max_tokens']==32
    packet=json.loads(requests[1]['messages'][-1]['content'])
    for key,value in json.loads(original[-1]['content']).items():assert packet[key]==value
    assert packet['output_recovery']['status']=='incomplete_unverified_model_draft'
    assert 'provider draft' in packet['output_recovery']['provider_returned_draft']
    usage=model.usage()
    assert usage['calls']==2 and usage['cost']==.00002 and model.reserved==0
    assert usage['output_recovery_attempts']==usage['recovered_structured_outputs']==1
    assert model.context_receipt['output_recovery']['draft_chars']>0


def test_a_second_truncation_stops_without_recursion(monkeypatch):
    model,requests=setup(monkeypatch,[truncated(),truncated()])
    with pytest.raises(OutputBudgetExceeded):asyncio.run(model(messages(),Turn,'medium'))
    assert len(requests)==2 and model.truncated_outputs==2
    assert model.output_recovery_attempts==1 and model.recovered_structured_outputs==0
    assert model.cost==.00002 and model.reserved==0


def test_recovery_allowance_is_shared_across_actor_calls(monkeypatch):
    model,requests=setup(monkeypatch,[truncated(),finished({'action':'finish','summary':'inspection'}),truncated()])
    asyncio.run(model(messages(),Turn,'medium'))
    with pytest.raises(OutputBudgetExceeded):asyncio.run(model(messages(),Turn,'medium'))
    assert len(requests)==3 and model.output_recovery_attempts==1


@pytest.mark.parametrize('condition',['disabled','mandatory','no_draft','critic','reasoning_off','existing_data'])
def test_ineligible_recovery_does_not_buy_another_request(monkeypatch,condition):
    class Critic(BaseModel): ready:bool
    model,requests=setup(monkeypatch,[truncated('' if condition=='no_draft' else 'private provider draft')])
    if condition=='disabled':model.config['output_recovery']=False
    if condition=='mandatory':model.reasoning_meta={'mandatory':True}
    packet=messages()
    if condition=='existing_data':packet[-1]['content']=json.dumps({'task':'inspect','output_recovery':'original user data'})
    with pytest.raises(OutputBudgetExceeded) as exc:
        asyncio.run(model(packet,Critic if condition=='critic' else Turn,'low' if condition=='reasoning_off' else 'medium'))
    assert len(requests)==1 and model.output_recovery_attempts==0
    assert 'private provider draft' not in str(exc.value)


def test_recovery_must_reserve_its_cost_inside_original_envelope(monkeypatch):
    model,requests=setup(monkeypatch,[truncated()],max_cost_usd=.001,mock_cost=.00099)
    with pytest.raises(BudgetExceeded):asyncio.run(model(messages(),Turn,'medium'))
    assert len(requests)==1 and model.cost==.00099 and model.reserved==0
    assert model.recovered_structured_outputs==0


def test_native_recovery_returns_only_the_fresh_complete_call(monkeypatch):
    from src.coding.orchestration import scoped_schema
    def tool(name,args,reason):
        return {'finish_reason':reason,'message':{'content':'','reasoning':'Unfinished draft of an action.',
            'tool_calls':[{'id':'call1','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}
    model,requests=setup(monkeypatch,[tool('write',{'path':'forbidden.py','content':'bad'},'length'),
        tool('read',{'path':'module.py'},'tool_calls')],action_protocol='tools')
    schema=scoped_schema(Turn,{'read','write'},can_delegate=False,can_plan=False)
    result=asyncio.run(model(messages(),schema,'medium'))
    assert result.tool=='read' and result.args=={'path':'module.py'} and len(requests)==2
    assert requests[0]['tools']==requests[1]['tools']


def test_provider_draft_is_bounded_and_does_not_invent_opaque_reasoning():
    draft=ProviderDraft(20)
    draft.add({'encrypted':'not readable'})
    assert draft.packet()==''
    draft.add('start'*20);draft.add('final fragment')
    assert len(draft.text)==20 and draft.packet().endswith('final fragment')
    assert 'omitted' in draft.packet()


def test_only_fresh_recovered_action_executes_and_checks_still_gate_completion(tmp_path,monkeypatch):
    replies=[finished({'action':'plan','plan':['Write expected.py and verify']}),
        truncated('Untrusted incomplete suggestion: write forbidden.py before finishing.'),
        finished({'action':'tool','tool':'write','args':{'path':'expected.py','content':'VALUE=1\n'}}),
        finished({'action':'tool','tool':'check','args':{'name':'unit'}}),
        finished({'action':'finish','summary':'Created expected.py and checked its value'})]
    model,requests=setup(monkeypatch,replies)
    hitl.set_work_mode('auto')
    result=asyncio.run(CodingHarness(tmp_path,model=model,config={'learn':False,
        'checks':{'unit':[sys.executable,'-B','-c','from expected import VALUE; assert VALUE==1']}}).run(
            'Create only expected.py with VALUE=1 and verify it. Never create forbidden.py.',reasoning='medium'))
    assert result['status']=='done' and result['checks_passed']==['unit']
    assert len(requests)==5 and result['calls']==5 and not (tmp_path/'forbidden.py').exists()
