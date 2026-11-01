import asyncio
import json

import httpx
import pytest

from src.coding.dialogue import ActorDialogue, AssistantCapture, active_dialogue, encoded_size
from src.coding.harness import CodingHarness, Turn
from src.coding.model import OpenRouterModel, OutputBudgetExceeded
from src.coding.native_tools import collect_calls
from src.coding.orchestration import scoped_schema


def assistant(name='edit', arguments=None, call_id='actual-call'):
    return {'role': 'assistant', 'content': None, 'reasoning': 'Continue the existing plan.',
            'tool_calls': [{'id': call_id, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments or {'path': 'app.py', 'old': '1', 'new': '2'})}}]}


def test_stream_retains_real_call_id_arguments_and_reasoning_blocks():
    calls = {}
    collect_calls(calls, [{'index': 0, 'id': 'real-id', 'function': {'name': 'edit', 'arguments': '{"path":'}}])
    collect_calls(calls, [{'index': 0, 'function': {'arguments': '"app.py","old":"1","new":"2"}'}}])
    capture = AssistantCapture()
    capture.add({'reasoning': 'duplicate raw', 'reasoning_details': [
        {'index': 0, 'type': 'reasoning.text', 'text': 'first ', 'signature': None}]})
    capture.add({'reasoning_details': [{'index': 0, 'text': 'second', 'signature': 'signature'},
                                     {'index': 1, 'type': 'reasoning.encrypted', 'data': 'opaque'}]})
    message = capture.finish(calls)
    assert message['tool_calls'][0]['id'] == 'real-id'
    assert json.loads(message['tool_calls'][0]['function']['arguments'])['new'] == '2'
    assert 'reasoning' not in message
    assert message['reasoning_details'] == [
        {'index': 0, 'type': 'reasoning.text', 'text': 'first second', 'signature': 'signature'},
        {'index': 1, 'type': 'reasoning.encrypted', 'data': 'opaque'}]


@pytest.mark.parametrize('details', [
    [{'type': 'reasoning.text', 'text': 'no index'}],
    [{'index': 1, 'type': 'reasoning.text'}, {'index': 0, 'type': 'reasoning.text'}],
    [{'index': 0, 'type': 'reasoning.text'}, {'index': 0, 'type': 'reasoning.encrypted'}],
    [{'index': 0, 'type': 'reasoning.text'}, {'index': 1, 'type': 'reasoning.text'}, {'index': 0, 'text': 'out of sequence'}],
])
def test_ambiguous_streamed_blocks_are_not_replayed(details):
    capture = AssistantCapture()
    capture.add({'reasoning_details': details})
    assert capture.finish({0: {'id': 'id', 'name': 'read', 'arguments': '{}'}}) is None


def test_capture_never_truncates_reasoning_or_synthesizes_ids():
    capture = AssistantCapture(limit=200)
    capture.add({'reasoning': 'x' * 201})
    assert capture.finish({0: {'id': 'id', 'name': 'read', 'arguments': '{}'}}) is None
    capture = AssistantCapture()
    details = [{'type': 'reasoning.encrypted', 'data': 'opaque', 'id': 'block'}]
    capture.add({'content': '', 'reasoning_details': details}, complete=True)
    assert capture.finish({0: {'name': 'read', 'arguments': '{}'}}) is None
    assert capture.finish({0: {'id': 'id', 'name': 'read', 'arguments': '{}'}})['reasoning_details'] == details


def test_history_only_acknowledged_outcomes_whole_pairs_and_current_constraints():
    dialogue = ActorDialogue(True, exchanges=2)
    message = assistant()
    dialogue.capture(message)
    state = [{'role': 'system', 'content': 'current policy'},
             {'role': 'user', 'content': '{"human_steering":["read only"],"verified":false}'}]
    assert dialogue.attach(state, 5000) == (state, [])
    dialogue.record({'ok': True, 'output': 'actually changed app.py'}, '17')
    message['reasoning'] = 'mutated elsewhere'
    transcript, ids = dialogue.attach(state, 5000)
    assert ids == ['actual-call']
    assert transcript[1]['reasoning'] == 'Continue the existing plan.'
    assert json.loads(transcript[2]['content']) == {
        'ok': True, 'output': 'actually changed app.py', 'historical': True, 'evidence_id': '17'}
    assert transcript[-1] == state[-1]
    assert dialogue.attach(state, encoded_size(state)) == (state, [])
    for i in range(3):
        dialogue.capture(assistant(call_id=str(i)))
        dialogue.record({'ok': False, 'output': 'denied'}, str(i))
    transcript, ids = dialogue.attach(state, 5000)
    assert ids == ['1', '2']
    assert all(json.loads(m['content'])['ok'] is False for m in transcript if m['role'] == 'tool')


def test_parallel_actors_steering_cancellation_and_unexecuted_calls_do_not_leak():
    async def scenario():
        left, right = ActorDialogue(True), ActorDialogue(True)
        async def model(messages, schema, effort):
            current = active_dialogue.get()
            current.capture(assistant(call_id=messages[0]['content']))
            await asyncio.sleep(0)
            assert active_dialogue.get() is current
            return 'returned'
        await asyncio.gather(*[d.generate(model, [{'content': label}], None, 'low', 0)
                              for d, label in [(left, 'left'), (right, 'right')]])
        left.record({'ok': True, 'output': 'left'})
        right.record({'ok': True, 'output': 'right'})
        assert left.history[0][0]['tool_calls'][0]['id'] == 'left'
        assert right.history[0][0]['tool_calls'][0]['id'] == 'right'
        await left.generate(model, [{'content': 'new steering'}], None, 'low', 1)
        assert left.history == [] and right.history
        async def cancelled(*args):
            active_dialogue.get().capture(assistant())
            raise asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await left.generate(cancelled, [], None, 'low', 1)
        left.record({'ok': True, 'output': 'must not be retained'})
        assert not left.history and active_dialogue.get() is None
    asyncio.run(scenario())


def mock_provider(monkeypatch, responses, payloads):
    import src.llm.llm as llm
    monkeypatch.setattr(llm, 'api_key', lambda: 'test-key')
    def handler(request):
        payloads.append(json.loads(request.content))
        message, finish = responses.pop(0)
        return httpx.Response(200, json={'choices': [{'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 30, 'cost': 0}})
    factory = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: factory(transport=httpx.MockTransport(handler), **kw))
    model = OpenRouterModel({'model': 'test', 'action_protocol': 'tools', 'max_input_usd_per_million': 0,
                             'max_output_usd_per_million': 0, 'cloud_context_tokens': 32768})
    model.ready = True
    return model


def test_adapter_sends_actual_edit_and_result_within_budget_without_replaying_execution(monkeypatch):
    payloads = []
    model = mock_provider(monkeypatch, [(assistant(), 'tool_calls'),
        (assistant('check', {'name': 'unit'}, 'next'), 'tool_calls')], payloads)
    dialogue = ActorDialogue(True)
    schema = scoped_schema(Turn, {'edit', 'check'}, False, False)
    messages = [{'role': 'system', 'content': 'policy'}, {'role': 'user', 'content': '{"task":"fix"}'}]
    async def scenario():
        await dialogue.generate(model, messages, schema, 'medium', 0)
        dialogue.record({'ok': True, 'output': 'edit succeeded'}, '5')
        turn = await dialogue.generate(model, messages, schema, 'medium', 0)
        assert turn.tool == 'check'
    asyncio.run(scenario())
    history = payloads[-1]['messages']
    assert history[2] == assistant()
    assert json.loads(history[-1]['content'])['evidence_id'] == '5'
    assert history[-1]['role'] == 'tool'
    assert json.loads(history[1]['content'])['task'] == 'fix'
    assert model.context_receipt['native_dialogue_ids'] == ['actual-call']
    assert model.context_receipt['input_utf8_bytes'] == encoded_size(history)
    assert encoded_size(history) + encoded_size({'tools': payloads[-1]['tools']}) + 8192 + 512 <= 32768


def test_continuation_preserves_current_state_and_steering_without_duplicate_read_text():
    dialogue=ActorDialogue(True)
    dialogue.capture(assistant('read',{'path':'app.py'}))
    dialogue.record({'ok':True,'output':'real current source'},'7')
    packet={'task':'Fix app','human_steering':[{'id':'u1','text':'Do not edit tests'}],
            'verified':False,'checks':{'unit':['python','-m','unittest']},
            'inspected_content':{'app.py':'real current source','other.py':'keep this'},
            'recent':[{'tool':'read','evidence_id':'7','output':'real current source'},
                      {'error':'An independent requirement is still unmet'}]}
    messages=[{'role':'system','content':'policy'}, {'role':'user','content':json.dumps(packet)}]
    transcript,ids=dialogue.attach(messages,5000)
    assert ids==['actual-call'] and transcript[-1]['role']=='tool'
    user=json.loads(transcript[1]['content'])
    assert user=={'task':'Fix app','human_steering':packet['human_steering']}
    last=json.loads(transcript[-1]['content'])
    assert last['output']=='real current source' and last['historical']
    assert last['current_state']['verified'] is False
    assert last['current_state']['checks']==packet['checks']
    assert last['current_state']['inspected_content']=={'other.py':'keep this'}
    assert last['current_state']['recent']==[packet['recent'][1]]
    assert json.loads(messages[-1]['content'])==packet
    # A changed file must retain its NEW content even when an older read is in history.
    packet['inspected_content']['app.py']='changed source'
    messages[-1]['content']=json.dumps(packet)
    current=json.loads(dialogue.attach(messages,5000)[0][-1]['content'])['current_state']
    assert current['inspected_content']['app.py']=='changed source'


@pytest.mark.parametrize('finish', ['length', 'tool_calls'])
def test_truncated_action_is_dropped_and_unknown_tool_is_only_recorded_as_rejected(monkeypatch, finish):
    payloads = []
    message = assistant('edit' if finish == 'length' else 'not_allowed')
    model = mock_provider(monkeypatch, [(message, finish)], payloads)
    dialogue = ActorDialogue(True)
    with pytest.raises((OutputBudgetExceeded, ValueError)):
        asyncio.run(dialogue.generate(model, [{'role': 'system', 'content': 'policy'},
            {'role': 'user', 'content': '{}'}], scoped_schema(Turn, {'edit'}), 'medium', 0))
    dialogue.record({'ok': True, 'output': 'must not be recorded'})
    if finish=='length':
        assert not dialogue.history
    else:
        assert len(dialogue.history)==1
        feedback=json.loads(dialogue.history[0][1]['content'])
        assert feedback['ok'] is False and 'nothing was executed' in feedback['output']
        assert 'capability set' in feedback['output'] and 'evidence_id' not in feedback


def test_malformed_native_arguments_cannot_break_context_or_enter_history():
    for args in ['not JSON','[]','null']:
        assert AssistantCapture().finish({0:{'id':'id','name':'read','arguments':args}}) is None


def test_rejected_repeat_returns_real_tool_feedback_without_execution(monkeypatch):
    payloads=[]
    model=mock_provider(monkeypatch,[(assistant('read',{'path':'app.py'},'rejected'), 'tool_calls'),
        (assistant('check',{'name':'unit'},'new-action'),'tool_calls')],payloads)
    dialogue=ActorDialogue(True)
    schema=scoped_schema(Turn,{'read','check'},False,False,
        blocked_calls=[{'tool':'read','args':{'path':'app.py'}}])
    messages=[{'role':'system','content':'policy'}, {'role':'user','content':'{"task":"fix"}'}]
    async def scenario():
        with pytest.raises(ValueError,match='Unchanged repeated call'):
            await dialogue.generate(model,messages,schema,'medium',0)
        assert dialogue.pending is None and len(dialogue.history)==1
        result=await dialogue.generate(model,messages,schema,'medium',0)
        assert result.tool=='check'
    asyncio.run(scenario())
    response=json.loads(payloads[-1]['messages'][-1]['content'])
    assert response['ok'] is False and 'nothing was executed' in response['output']
    assert 'Unchanged repeated call' in response['output']
    assert 'evidence_id' not in response


def test_root_runtime_acknowledges_real_mutation_but_completion_still_requires_checks(tmp_path, monkeypatch):
    from src.runtime import hitl
    monkeypatch.setattr(hitl, '_work_mode', {'': 'auto'})
    (tmp_path / 'app.py').write_text('VALUE=1\n')
    seen = []
    turns = [dict(action='plan', plan=['Fix value and test']),
             dict(action='tool', tool='edit', args={'path': 'app.py', 'old': '1', 'new': '2'}),
             dict(action='finish', summary='done')]
    async def model(messages, schema, effort):
        dialogue = active_dialogue.get()
        seen.append(list(dialogue.history))
        action = turns.pop(0)
        name = action.get('tool', action['action'])
        args = action.get('args', {k: v for k, v in action.items() if k != 'action'})
        dialogue.capture(assistant(name, args, 'call-' + str(len(seen))))
        return schema.model_validate(action)
    result = asyncio.run(CodingHarness(tmp_path, model=model,
        config={'max_turns': 3, 'native_history': True, 'checks': {}}).run('Set VALUE=2'))
    assert (tmp_path / 'app.py').read_text() == 'VALUE=2\n'
    assert result['status'] != 'done'
    assert len(seen[-1]) == 2
    outcome = json.loads(seen[-1][-1][1]['content'])
    assert outcome['ok'] is True and outcome['evidence_id'].isdigit()
