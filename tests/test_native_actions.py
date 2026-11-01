import asyncio
import json

import httpx
import pytest

from src.coding.harness import Turn
from src.coding.model import OpenRouterModel
from src.coding.native_tools import compile_tools,collect_calls,decode_call
from src.coding.orchestration import scoped_schema


@pytest.mark.parametrize('focused',[False,True])
@pytest.mark.parametrize('child',[False,True])
def test_native_prompt_has_one_protocol_and_keeps_authority_and_completion_rules(focused,child):
    from src.coding.prompts import system_prompt,action_policy,HINTS
    schema=scoped_schema(Turn,{'read','edit','check'},False,False,registered_checks={'unit':[]})
    text=system_prompt({'read','edit','check'},'PROFILE: preserve this task strategy.',
                       child=child,domains=['software'],focused=focused,native=True)+action_policy(schema,native=True)
    assert 'Invoke exactly one provided function' in text
    assert 'Return one JSON action' not in text and 'Return one available JSON action' not in text
    assert 'Tool actions use action=' not in text and 'tool {tool,args' not in text
    assert 'Operator' in text and 'untrusted data' in text and 'registered checks' in text
    assert 'PROFILE: preserve this task strategy.' in text and 'finish.summary' in text
    if child:assert 'SUBAGENT:' in text and 'complete assignment' in text
    assert HINTS['read'] not in text  # Single authoritative definition is in the native tool list.
    definition=next(d['function'] for d in compile_tools(schema) if d['function']['name']=='read')
    assert definition['description']==HINTS['read']
    assert 'Return one' in system_prompt({'read'},focused=focused,native=False)


@pytest.mark.parametrize('orchestration',[False,True])
def test_native_root_preserves_environment_contract_and_exact_task(tmp_path,orchestration):
    from src.coding.harness import CodingHarness,SYSTEM
    class Model:
        async def __call__(self,messages,schema,effort):
            text=messages[0]['content']
            assert 'Invoke exactly one provided function' in text
            assert 'Return one JSON action' not in text
            assert 'tool {tool,args' not in text
            assert json.loads(messages[-1]['content'])['task']=='Inspect without changes.'
            return schema.model_validate({'action':'finish','summary':'No changes requested.'})
    result=asyncio.run(CodingHarness(tmp_path,model=Model(),config={
        'backend':'openrouter','action_protocol':'tools','learn':False,
        'orchestration':{'enabled':orchestration,'same_model':True,'critic':{'enabled':False}}}
    ).run('Inspect without changes.'))
    assert result['status']=='done'


def test_registered_check_names_are_bound_in_json_native_and_runtime():
    schema=scoped_schema(Turn,{'check','read'},registered_checks={'unit': ['python','-m','unittest']})
    tools=compile_tools(schema)
    check=next(d['function'] for d in tools if d['function']['name']=='check')
    assert check['parameters']['properties']['name']['enum']==['unit']
    assert schema.model_validate({'action':'tool','tool':'check','args':{'name':'unit'}}).tool=='check'
    with pytest.raises(ValueError,match='exact registered check'):
        schema.model_validate({'action':'tool','tool':'check','args':{'name':'test_module.py'}})
    assert 'check' not in {d['function']['name'] for d in compile_tools(
        scoped_schema(Turn,{'check','read'},registered_checks={}))}
    # A registration updates the next step's schema, without changing other calls.
    next_schema=scoped_schema(Turn,{'check','read'},registered_checks={'unit':[],'lint':[]})
    assert next_schema.model_validate({'action':'tool','tool':'check','args':{'name':'lint'}})
    assert schema.model_validate({'action':'tool','tool':'read','args':{'path':'test_module.py'}})


def test_native_tools_preserve_per_step_action_and_tool_permissions():
    schema=scoped_schema(Turn,{'read','math'},can_delegate=False,can_plan=False)
    definitions=compile_tools(schema)
    assert {d['function']['name'] for d in definitions}=={'read','math','finish'}
    calls={}
    collect_calls(calls,[{'index':0,'function':{'name':'read','arguments':'{"pa'}}])
    collect_calls(calls,[{'index':0,'function':{'arguments':'th":"module.py"}'}}])
    turn=Turn.model_validate(decode_call(calls,definitions))
    assert turn.action=='tool' and turn.tool=='read' and turn.args=={'path':'module.py'}
    calls[0]['name']='edit'
    with pytest.raises(ValueError,match='capability'):decode_call(calls,definitions)


def test_native_protocol_never_executes_multiple_or_missing_calls():
    definitions=compile_tools(Turn)
    with pytest.raises(ValueError,match='exactly ONE'):decode_call({},definitions)
    with pytest.raises(ValueError,match='exactly ONE'):
        decode_call({0:{'name':'read','arguments':'{}'},1:{'name':'write','arguments':'{}'}},definitions)
    with pytest.raises(ValueError,match='object'):
        decode_call({0:{'name':'read','arguments':'[]'}},definitions)


def test_native_stream_is_decoded_validated_and_billed_on_the_same_adapter(monkeypatch):
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            chunks=[{'provider':'TestEndpoint','choices':[{'delta':{'tool_calls':[
                {'index':0,'function':{'name':'read','arguments':'{"path":'}}]}}]},
                {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"app.py"}'}}]}}]},
                {'choices':[],'usage':{'prompt_tokens':100,'completion_tokens':20,'cost':.00002}}]
            for chunk in chunks: yield ('data: '+json.dumps(chunk)+'\n\n').encode()
    def handler(request):
        data=json.loads(request.content)
        assert 'response_format' not in data and data['tool_choice']=='auto'
        assert 'parallel_tool_calls' not in data
        assert {d['function']['name'] for d in data['tools']}=={'read','finish'}
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Stream())
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handler),**kw))
    model=OpenRouterModel({'model':'cheap','action_protocol':'tools'});model.ready=True
    result=asyncio.run(model([{'role':'system','content':'policy'},{'role':'user','content':'{"task":"inspect"}'}],
                            scoped_schema(Turn,{'read'},False,False),'low'))
    assert result.tool=='read' and model.usage()['cost']==.00002 and model.reserved==0
    assert model.usage()['endpoints_used']=={'TestEndpoint':1}
    assert model.context_receipt['action_protocol']=='tools'


def test_native_provider_without_tool_choice_does_not_receive_unsupported_parameter(monkeypatch):
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    def handler(request):
        payload=json.loads(request.content)
        assert 'tools' in payload and 'tool_choice' not in payload
        assert payload['provider']['require_parameters'] is True
        return httpx.Response(200,json={'choices':[{'message':{'tool_calls':[{
            'id':'read-1','type':'function','function':{'name':'read','arguments':'{"path":"app.py"}'}}]},
            'finish_reason':'tool_calls'}],'usage':{'prompt_tokens':10,'completion_tokens':10,'cost':0}})
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handler),**kw))
    model=OpenRouterModel({'model':'native-only','action_protocol':'tools'});model.ready=True
    model.supported_parameters=['tools','reasoning','max_tokens']
    turn=asyncio.run(model([{'role':'system','content':'policy'},{'role':'user','content':'{}'}],
        scoped_schema(Turn,{'read'},False,False),'medium'))
    assert turn.tool=='read'


def test_native_history_uses_observations_preserving_arguments_and_constraints():
    from src.coding.native_tools import observation_messages
    from src.coding.request import compact_packet
    packet={'task':'Compute without changing files','human_steering':[{'id':'1','text':'Keep exact fractions'}],
        'inspected_content':{'app.py':'real source'},'recent':[
            {'tool':'read','args':{'path':'app.py'},'ok':True,'output':'real source','evidence_id':'3'},
            {'tool':'math','args':{'op':'sum','expr':'k','var':'k','bounds':[1,5]},
             'ok':False,'output':'recorded failure','evidence_id':'4'}]}
    messages=[{'role':'system','content':'policy'},{'role':'user','content':json.dumps(compact_packet(packet))}]
    transcript,ids=observation_messages(messages,compile_tools(Turn))
    assert ids==['3','4']
    assert json.loads(transcript[2]['content'])=={'ok':True,'output':'real source','evidence_id':'3'}
    assert json.loads(transcript[3]['tool_calls'][0]['function']['arguments'])['bounds']==[1,5]
    assert json.loads(transcript[4]['content'])['ok'] is False
    state=json.loads(transcript[-1]['content'])
    assert not state['recent'] and state['inspected_content']==packet['inspected_content']
    assert state['task']==packet['task'] and state['human_steering']==packet['human_steering']
    assert len(json.loads(messages[-1]['content'])['recent'])==2


def test_native_history_never_invents_calls_from_clipped_or_unavailable_observations():
    from src.coding.native_tools import observation_messages
    packet={'working_memory':{'observations':[
        {'tool':'read','args':{'path':'app.py'},'ok':True,'output':'x','evidence_id':'1'},
        {'tool':'math','args':{'expr':'k [omitted; use evidence]'},'ok':True,'output':'x','evidence_id':'2'},
        {'tool':'math','args':{},'args_truncated':True,'ok':True,'output':'x','evidence_id':'3'},
        {'tool':'math','args':{'op':'sum'},'ok':True,'output':'x'}]}}
    messages=[{'role':'user','content':json.dumps(packet)}]
    assert observation_messages(messages,compile_tools(scoped_schema(Turn,{'math'})))==(messages,[])
