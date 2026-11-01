import asyncio
import json

import pytest

from src.agent_train.trajectory import TrajectoryLog, read_events
from src.agent_train.dataset import inspect_trace, export


def sample(tmp_path, *, split='train', status='done', reward=True):
    log = TrajectoryLog(tmp_path, {'task': 'Fix a module', 'model': 'test', 'harness': {'source_id': 'abc'}})
    log.model_event('main', 'request', {'messages': [{'role': 'user', 'content': 'Fix'}],
        'schema': {'type': 'object'}, 'generation': {'model': 'test'}})
    log.model_event('main', 'response', {'content': '{"action":"tool","tool":"read","args":{"path":"a.py"}}',
                                       'finish_reason': 'stop'})
    log.model_event('main', 'validated', {'action': 'tool'})
    (tmp_path / '1.json').write_text('{"ok":true,"output":"VALUE=2"}')
    log.action('main', 'read', True, '1', tmp_path)
    log.outcome({'status': status})
    log.evaluation(task_id='a-1', family='a', split=split, verifier_id='external-test-v1',
        answer_ok=reward, process_ok=reward, status=status, observation={'ok': reward}, eligible=True)
    return log


def test_interleaved_actors_have_exact_context_and_separate_call_ids(tmp_path):
    log = TrajectoryLog(tmp_path, {})
    log.model_event('main', 'request', {'messages': ['parent'], 'schema': {'oneOf': []}})
    log.model_event('child', 'request', {'messages': ['child'], 'schema': {'type': 'object'}})
    log.model_event('main', 'response', {'content': 'parent response'})
    log.model_event('child', 'response', {'content': 'child response'})
    log.model_event('main', 'reasoning', 'not a supervised target')
    events = read_events(log.path)
    assert len(events) == 5
    assert events[1]['call_id'] == events[3]['call_id']
    assert events[2]['call_id'] == events[4]['call_id'] != events[1]['call_id']
    assert events[1]['data']['messages'] == ['parent']
    assert log.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('split,status,reward', [('diagnostic', 'done', True), ('test', 'done', True),
    ('train', 'paused', True), ('train', 'done', False)])
def test_unverified_partial_and_holdout_runs_never_enter_sft(tmp_path, split, status, reward):
    log = sample(tmp_path, split=split, status=status, reward=reward)
    assert not inspect_trace(log.path)['eligible']
    assert export([log.path], tmp_path / 'dataset')['sft_rows'] == 0


def test_only_executed_validated_steps_from_external_success_are_exported(tmp_path):
    log = sample(tmp_path)
    report = export([log.path, log.path], tmp_path / 'dataset')
    assert report['sft_rows'] == 1 and not report['grpo_ready']
    row = json.loads((tmp_path / 'dataset' / 'train.jsonl').read_text())
    assert len(row['messages']) == 2 and row['messages'][-1]['role'] == 'assistant'
    # A later proposal with no execution cannot become a supervised action.
    log.model_event('main', 'request', {'messages': [{'role': 'user', 'content': 'stale'}], 'schema': {}})
    log.model_event('main', 'validated', {'action': 'tool'})
    assert export([log.path], tmp_path / 'incomplete')['sft_rows'] == 0


def test_redaction_and_full_log_limit_are_excluded(tmp_path, monkeypatch):
    monkeypatch.setenv('DEMO_API_KEY', 'privatevalue12345')
    log = sample(tmp_path)
    log.record('extra', {'api_key': 'something', 'source': 'privatevalue12345'})
    assert not inspect_trace(log.path)['eligible']
    assert all('privatevalue12345' not in p.read_text() for p in log.directory.rglob('*.json'))
    small = TrajectoryLog(tmp_path, {'task': 'x' * 500}, max_bytes=100)
    assert not small.receipt()['complete'] and small.receipt()['error']


def test_corruption_fails_closed(tmp_path):
    log = sample(tmp_path)
    blob = next((log.directory / 'blobs').glob('*.json'))
    blob.write_text('{}')
    with pytest.raises(ValueError, match='Corrupt'): read_events(log.path)
    assert export([log.path], tmp_path / 'dataset')['sft_rows'] == 0


def test_done_by_itself_never_assigns_reward(tmp_path):
    log = TrajectoryLog(tmp_path, {})
    log.outcome({'status': 'done', 'checks_passed': ['unit']})
    assert 'no_external_verifier' in inspect_trace(log.path)['excluded']
    log.evaluation(task_id='x', family='x', split='train', verifier_id='v1', answer_ok=None,
        process_ok=False, status='error', observation={'error': 'provider failed'}, eligible=True)
    assert read_events(log.path)[-1]['data']['reward'] is None
    log.evaluation(task_id='x', family='x', split='train', verifier_id='v1', answer_ok=False,
        process_ok=False, status='paused', observation={'error': '429'}, eligible=True, infrastructure_error=True)
    assert read_events(log.path)[-1]['data']['reward'] is None


def test_accepted_finish_is_a_supervised_action_but_rejected_finish_is_not(tmp_path):
    log = TrajectoryLog(tmp_path, {'model': 'test'})
    for accepted in (False, True):
        log.model_event('main', 'request', {'messages': [{'role': 'user', 'content': str(accepted)}],
                                          'schema': {'type': 'object'}})
        log.model_event('main', 'response', {'content': '{"action":"finish","summary":"done"}'})
        log.model_event('main', 'validated', {'action': 'finish'})
        if accepted: log.record('finish', {'status': 'done'}, 'main', log.calls['main'])
    log.outcome({'status': 'done'})
    log.evaluation(task_id='x', family='x', split='train', verifier_id='external', answer_ok=True,
        process_ok=True, status='done', observation={'ok': True}, eligible=True)
    report = export([log.path], tmp_path/'dataset')
    assert report['sft_rows'] == 1
    assert json.loads((tmp_path/'dataset/train.jsonl').read_text())['messages'][0]['content'] == 'True'


def test_native_call_capture_keeps_actual_wire_schema_and_arguments(tmp_path, monkeypatch):
    import httpx
    from src.coding.model import OpenRouterModel
    from src.coding.harness import Turn
    import src.llm.llm as llm
    monkeypatch.setattr(llm, 'api_key', lambda: 'test-key')
    sent = []
    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'data': [{'id': 'small', 'supported_parameters': ['tools'],
                'pricing': {'prompt': '0', 'completion': '0'}}]})
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'tool_calls', 'message': {
            'content': None, 'tool_calls': [{'id': 'actual-id', 'type': 'function',
                'function': {'name': 'finish', 'arguments': '{"summary":"observed"}'}}]}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'cost': 0}})
    factory = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: factory(transport=httpx.MockTransport(handler), **kw))
    model = OpenRouterModel({'model': 'small', 'action_protocol': 'tools', 'cloud_context_tokens': 32768})
    log = TrajectoryLog(tmp_path, {})
    model.on_event = lambda kind, data: log.model_event('main', kind, data)
    asyncio.run(model([{'role': 'system', 'content': 'policy'}, {'role': 'user', 'content': '{"task":"inspect"}'}], Turn, 'low'))
    events = read_events(log.path)
    request = next(e['data'] for e in events if e['kind'] == 'request')
    response = next(e['data'] for e in events if e['kind'] == 'response')
    assert request['messages'] == sent[0]['messages']
    assert request['generation']['tools'] == sent[0]['tools']
    assert 'Authorization' not in str(request)
    assert response['tool_calls'][0]['arguments'] == '{"summary":"observed"}'
    assert response['tool_calls'][0]['id'] == 'actual-id'


def test_harness_emits_outcome_without_assuming_external_success(tmp_path):
    from src.coding.harness import CodingHarness
    class Model:
        async def __call__(self, messages, schema, effort):
            return schema.model_validate({'action': 'finish', 'summary': 'Inspected; no changes'})
    h = CodingHarness(tmp_path, model=Model(), config={'max_turns': 1, 'learn': False})
    result = asyncio.run(h.run('Explain the empty project'))
    assert result['trajectory']['complete']
    events = read_events(result['trajectory']['path'])
    assert events[0]['data']['harness']['source_id']
    assert events[-1]['kind'] == 'outcome'
    assert 'no_external_verifier' in inspect_trace(result['trajectory']['path'])['excluded']
