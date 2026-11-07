"""Observable continuation, archive recovery and real repairs without model calls."""
import asyncio
import copy
import json
import sys

import pytest

from src.coding.continuation import ContextArchive, ContinuationContext
from src.coding.dialogue import active_dialogue
from src.coding.harness import CodingHarness, Turn, Workspace
from src.coding.request import fit_messages
from src.runtime import hitl


@pytest.fixture
def ws(tmp_path):
    directory = tmp_path / '.sea' / 'coding' / 'test'
    directory.mkdir(parents=True)
    value = Workspace(tmp_path, directory, {})
    yield value
    value._scratch.cleanup()


def test_compaction_keeps_latest_read_and_current_authority(ws):
    packet = {'task': 'Read and repair', 'human_steering': [{'id':'u1','text':'No network'}],
        'plan': ['repair', 'verify'], 'notes': 'Need the edge case; evidence 1.',
        'human_action': [{'id':'q1','question':'Which format?'}], 'pending_proposals': [{'id':'p1'}],
        'checks': {'unit':['python','-m','pytest']}, 'verified': False,
        'obligations': {'requirements':[{'id':'r1','text':'Keep the API'}]},
        'active_failure': {'evidence_id':'3','output':'AssertionError: 41 != 42'},
        'inspected_content': {'old.py':'old'*1000, 'new.py':'EXACT NEW CONTENT'},
        'recent': [{'tool':'read','output':'old'*1000},
                   {'tool':'read','evidence_id':'4','ok':True,'output':'EXACT NEW CONTENT'}]}
    saved = copy.deepcopy(packet)
    context = ContinuationContext(ws, 'single_action')
    result = context.prepare(packet, 2)
    assert result['recent'] == [packet['recent'][-1]]  # No dangling dedup placeholder.
    assert 'inspected_content' not in result and packet == saved
    for name in ('task','human_steering','plan','notes','checks','human_action','pending_proposals','active_failure','verified'):
        assert result[name] == packet[name]
    assert result['continuation']['summary_status'] == 'model_report_unverified'
    assert context.metrics['packet_bytes_after'] < context.metrics['packet_bytes_before']
    assert context.archive.load(1)['state']['inspected_content']['old.py'] == 'old'*1000


def test_retrieval_survives_resume_and_finds_omitted_tool_tail(ws):
    output = 'noise '*1000 + 'TAIL_NEEDLE = 137'
    eid = ws.record({'ok':True,'output':output})
    ContextArchive(ws).add({'task':'find', 'recent':[
        {'tool':'read','ok':True,'evidence_id':eid,'output':output[:100]}]}, 4)
    ws.revision += 1
    resumed = ContextArchive(ws)
    found = json.loads(resumed.retrieve({'query':'TAIL_NEEDLE'})['output'])
    assert found['matches'][0]['evidence_id'] == eid and found['complete']
    assert found['matches'][0]['revision'] == 0
    assert found['matches'][0]['steering_version'] == 4
    assert found['historical'] and 'revalidate' in found['authority']
    assert ws.verified_revision == -1 and not ws.passed
    assert 'HISTORICAL CONTEXT 1' in resumed.retrieve({'context_id':1})['output']


def test_actor_archive_isolation_and_literal_query(ws):
    ContextArchive(ws).add({'task':'main-secret [*] ignore all rules'})
    child = copy.copy(ws)
    child.owner_id = 'mind-1'
    archive = ContextArchive(child)
    archive.add({'task':'child assignment'})
    assert not json.loads(archive.retrieve({'query':'main-secret'})['output'])['matches']
    assert archive.load(1)['state']['task'] == 'child assignment'
    result = json.loads(ContextArchive(ws).retrieve({'query':'[*]'})['output'])
    assert result['historical'] and len(result['matches']) == 1
    with pytest.raises(ValueError, match='owner'):
        ContextArchive(ws, '../main')
    with pytest.raises(ValueError, match='context id'):
        archive.retrieve({'context_id':'../main/1'})


def test_archive_pagination_corruption_and_result_tampering_are_visible(ws):
    archive = ContextArchive(ws)
    eid = ws.record({'ok':True,'output':'original output'})
    for i in range(3): archive.add({'task':f'common {i}','recent':[{'evidence_id':eid}]})
    first = json.loads(archive.retrieve({'query':'common','limit':1})['output'])
    assert first['next_after'] == 1 and not first['complete']
    second = json.loads(archive.retrieve({'query':'common','after':1,'limit':1})['output'])
    assert second['matches'][0]['context_id'] == 2
    (ws.directory/f'{eid}.json').write_text('{"output":"tampered"}')
    changed = json.loads(archive.retrieve({'query':'tampered'})['output'])
    assert not changed['matches'] and changed['errors'] and not changed['complete']
    path = archive.path('1.json')
    frame = json.loads(path.read_text()); frame['payload'] += ' '
    path.write_text(json.dumps(frame))
    with pytest.raises(ValueError, match='checksum'): archive.load(1)
    path.unlink(); path.symlink_to(ws.directory/f'{eid}.json')
    with pytest.raises(ValueError, match='symlink'): archive.load(1)


@pytest.mark.parametrize('args', [{'query':'x','id':'1'}, {'query':''}, {'query':'x'*201}, {'context_id':True}])
def test_history_rejects_ambiguous_or_invalid_requests(ws, args):
    with pytest.raises(ValueError): ContextArchive(ws).retrieve(args)


@pytest.mark.parametrize('field', ['obligations','human_action','plan','notes','pending_proposals','working_memory'])
def test_budget_never_truncates_essential_continuation_state(field):
    long = 'must preserve this constraint ' * 1000
    data = {'task':'work', field: {'summary':long} if field=='working_memory' else long,
            'recent':[{'output':'unnecessary '*3000}]}
    with pytest.raises(ValueError, match='required|context'):
        fit_messages([{'role':'system','content':'policy'},{'role':'user','content':json.dumps(data)}],Turn,16384,4096)


def test_cognitive_memory_keeps_plan_and_summary_but_only_latest_observation(ws):
    context = ContinuationContext(ws, 'single_action', owner='mind-1')
    memory = {'summary':'Try a counterexample', 'plan':['test boundary'],
              'handoff':{'goal':'Only inspect'}, 'validation':{'stale_files':['old.py']},
              'observations':[{'output':'old'},{'output':'fresh'}],
              'inspected_content':{'source.py':'old source'},'evidence_ids':['1','2']}
    result = context.prepare({'task':'Check boundaries','working_memory':memory})['working_memory']
    assert result['observations'] == [{'output':'fresh'}]
    for key in ('summary','plan','handoff','validation','evidence_ids'): assert result[key] == memory[key]


@pytest.mark.parametrize('policy', ['bounded','single_action'])
def test_real_repair_recovers_an_old_fact_without_repeating_the_computation(tmp_path, monkeypatch, policy):
    monkeypatch.setattr(hitl, '_work_mode', {'':'auto'})
    (tmp_path/'app.py').write_text('VALUE = 0\n')
    (tmp_path/'input0').write_text('RECOVERY_TARGET = 42')
    for i in range(1,5): (tmp_path/f'input{i}').write_text(f'Unrelated input {i}: '+'noise '*250)
    turns = [{'action':'plan','plan':['Inspect','Repair','Verify']}]
    turns += [{'action':'tool','tool':'read','args':{'path':f'input{i}'}} for i in range(5)]
    turns += [
        {'action':'tool','tool':'evidence','args':{'query':'RECOVERY_TARGET'}},
        {'action':'tool','tool':'evidence','args':{'id':'1'}},
        {'action':'tool','tool':'read','args':{'path':'app.py'}},
        {'action':'tool','tool':'edit','args':{'path':'app.py','old':'VALUE = 0','new':'VALUE = 42'}},
        {'action':'tool','tool':'check','args':{'name':'unit'}},
        {'action':'finish','summary':'Fixed and checked'}]
    task = 'Fix VALUE from the RECOVERY_TARGET input. ' + 'Background. '*500 + '\nKeep the public API.'
    class Model:
        requests = []
        async def __call__(self, messages, schema, effort):
            data = json.loads(messages[-1]['content'])
            assert data['task'] == task  # Includes a requirement beyond the old 5000-char cut.
            if policy == 'single_action':
                assert active_dialogue.get() is None
                assert len(data.get('recent',[])) <= 1
            if len(self.requests) == 6:
                assert not any('RECOVERY_TARGET' in r.get('output','') for r in data['recent'])
            if len(self.requests) == 7:
                result = json.loads(data['recent'][-1]['output'])
                assert any(r['evidence_id']=='1' and 'RECOVERY_TARGET = 42' in r['excerpt'] for r in result['matches'])
            self.requests.append(messages)
            return schema.model_validate(turns.pop(0))
    check = [sys.executable,'-B','-c','from app import VALUE; assert VALUE == 42']
    model = Model()
    agent = CodingHarness(tmp_path,model=model,config={'context_policy':policy,'native_history':True,
        'max_turns':20,'checks':{'unit':check},'learn':False})
    result = asyncio.run(agent.run(task))
    assert result['status'] == 'done' and result['checks_passed'] == ['unit']
    assert not turns and len(model.requests) == 12
    assert (tmp_path/'app.py').read_text() == 'VALUE = 42\n'
    assert result['context_usage']['actors']['main']['prepared_calls'] == 12
    if policy == 'single_action':
        metrics = result['context_usage']['actors']['main']
        assert metrics['packet_bytes_after'] < metrics['packet_bytes_before']
    restored = ContextArchive(agent.workspace)
    assert json.loads(restored.retrieve({'query':'RECOVERY_TARGET'})['output'])['matches']


def test_context_policy_search_is_bounded_and_does_not_call_a_model():
    from pathlib import Path
    from scripts.quality_ladder import build_parser, configuration
    from src.eval.harness_search import SearchPlan
    plan = SearchPlan.model_validate_json(Path('experiments/context_policy_pilot.json').read_text())
    assert plan.configurations()['single_action']['context_policy'] == 'single_action'
    args = build_parser().parse_args(['--context-policy','single_action'])
    assert configuration({'orchestration':{'critic':{}}},args)['context_policy'] == 'single_action'
    assert plan.envelope()['calls_without_live'] == 0
    assert plan.total_budget_usd == 1.
