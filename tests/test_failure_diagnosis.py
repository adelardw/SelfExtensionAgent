"""Failure recovery never authorizes completion or silently patches source/tests."""
import asyncio
import json
import sys

import pytest

from src.coding.control import SessionControl
from src.coding.critic import ContextCritic
from src.coding.failure_diagnosis import diagnose_failure
from src.coding.harness import CodingHarness, Workspace
from src.coding.minds import MindScheduler
from src.coding.supervisor import Supervisor
from src.runtime import hitl


class DiagnosisModel:
    def __init__(self, hook=None):
        self.calls = 0
        self.hook = hook

    async def __call__(self, messages, schema, effort):
        self.calls += 1
        assert schema.__name__ == 'FailureDiagnosis' and effort == 'low'
        data = json.loads(messages[-1]['content'])
        wire = schema.model_json_schema()
        assert wire['properties']['evidence']['items']['enum'] == data['valid_evidence_ids']
        assert wire['properties']['next_steps']['items']['maxLength'] == 300
        if self.hook: self.hook()
        return schema.model_validate({'category': 'test_expectation',
            'reason': 'The specification doubles x, so double(3) is 6, not 5.',
            'evidence': [data['failure']['evidence_id']],
            'next_steps': ['Correct the new expectation and rerun the registered check.']})


def setup(tmp_path, model=None, max_calls=8):
    directory = tmp_path / '.sea'; directory.mkdir(exist_ok=True)
    ws = Workspace(tmp_path, directory, {'unit': [sys.executable, '-B', 'test_double.py']})
    (tmp_path / 'numbers_impl.py').write_text('def double(x): return 2*x\n')
    (tmp_path / 'test_double.py').write_text('from numbers_impl import double\nassert double(3)==5\n')
    ws.changed.add('test_double.py')
    control, model = SessionControl(), model or DiagnosisModel()
    supervisor = Supervisor(directory, control, model, False)
    control.supervisor = supervisor
    node = MindScheduler(control, supervisor, {}, ws).nodes['main']
    node.task = 'Add regression tests for double(x), which returns twice x, and run checks'
    node.memory.files['numbers_impl.py'] = ws.file_digest('numbers_impl.py')
    critic = ContextCritic(model, supervisor, control, {'enabled': True, 'max_calls': max_calls})
    return critic, node, ws


def test_diagnosis_is_advisory_bounded_and_persists_across_resume(tmp_path):
    hitl.set_work_mode('auto')
    async def run():
        critic, node, ws = setup(tmp_path)
        failed = await ws.execute('check', {'name': 'unit'})
        assert not failed['ok'] and 'AssertionError' in failed['output']
        eid = ws.record(failed)
        before = ws.file_digest('numbers_impl.py')
        report = await diagnose_failure(critic, node, ws, 'unit', eid)
        assert report['status'] == 'unverified_hypothesis'
        assert not {'ready', 'aligned', 'profile', 'tools'} & report.keys()
        assert ws.passed == set() and ws.verified_revision == -1
        assert ws.file_digest('numbers_impl.py') == before
        assert json.loads((ws.directory / (report['evidence_id']+'.json')).read_text())['critic_source']
        # More edits and a new critic instance must not buy another diagnosis.
        ws.revision += 1
        resumed = ContextCritic(critic.model, critic.supervisor, critic.control, critic.config)
        assert await diagnose_failure(resumed, node, ws, 'unit', eid) is None
        assert critic.model.calls == 1 and resumed.calls == 0
    asyncio.run(run())


@pytest.mark.parametrize('change', ['source', 'steering'])
def test_stale_diagnosis_is_discarded(tmp_path, change):
    async def run():
        critic, node, ws = setup(tmp_path)
        if change == 'source':
            critic.model.hook = lambda: (tmp_path/'numbers_impl.py').write_text('def double(x): return 3*x\n')
        else:
            critic.model.hook = lambda: critic.control.post('Preserve all tests', interrupt=False)
        eid = ws.record({'ok': False, 'output': 'AssertionError'})
        assert await diagnose_failure(critic, node, ws, 'unit', eid) is None
        assert critic.calls == 1
    asyncio.run(run())


def test_failure_recovery_reserves_verification_calls(tmp_path):
    critic, node, ws = setup(tmp_path, max_calls=3)
    critic.calls = 1
    eid = ws.record({'ok': False, 'output': 'AssertionError'})
    assert asyncio.run(diagnose_failure(critic, node, ws, 'unit', eid)) is None
    assert critic.model.calls == 0 and critic.calls == 1


def test_failure_recovery_also_reserves_an_unextracted_contract(tmp_path):
    critic, node, ws = setup(tmp_path, max_calls=3)
    eid = ws.record({'ok': False, 'output': 'AssertionError'})
    assert asyncio.run(diagnose_failure(critic, node, ws, 'unit', eid)) is None
    assert critic.calls == 0


def test_bad_diagnosis_is_not_retried_or_accepted(tmp_path):
    class Invalid(DiagnosisModel):
        async def __call__(self, *args):
            value = await super().__call__(*args)
            value.evidence = ['999999']
            return value
    async def run():
        critic, node, ws = setup(tmp_path, Invalid())
        eid = ws.record({'ok': False, 'output': 'AssertionError'})
        with pytest.raises(ValueError, match='evidence'):
            await diagnose_failure(critic, node, ws, 'unit', eid)
        assert await diagnose_failure(critic, node, ws, 'unit', eid) is None
        assert critic.calls == 1 and not ws.passed
    asyncio.run(run())


@pytest.mark.parametrize('tamper', [False, True])
def test_diagnostic_probe_adds_real_observations_without_replacing_check(tmp_path,tamper):
    class Executable(DiagnosisModel):
        async def __call__(self,*args):
            result=await super().__call__(*args)
            result.probe_code=(
                'from pathlib import Path\nPath("numbers_impl.py").write_text("def double(x): return 5\\n")\n'
                if tamper else '')+'from numbers_impl import double\nprint("double(3):", double(3))\n'
            return result
    async def run():
        hitl.set_work_mode('auto')
        critic,node,ws=setup(tmp_path,Executable())
        critic.config['active']=True
        before=ws.file_digest('numbers_impl.py')
        eid=ws.record({'ok':False,'output':'AssertionError: expected 5, observed 6'})
        report=await diagnose_failure(critic,node,ws,'unit',eid)
        probe=report['diagnostic_observations'][0]
        assert probe['ok'] is not tamper
        assert ('changed original snapshot files' if tamper else 'double(3): 6') in probe['output']
        assert ws.file_digest('numbers_impl.py')==before
        assert not ws.passed and ws.verified_revision==-1 and critic.calls==1
        assert report['status']=='unverified_hypothesis'
    asyncio.run(run())


def test_actor_gets_observations_without_the_unexecuted_prior_opinion():
    from src.coding.failure_diagnosis import diagnostic_packet
    report={'status':'unverified_hypothesis','category':'implementation','reason':'A prior opinion',
            'next_steps':['Rewrite the implementation'],
            'diagnostic_observations':[{'id':'4','output':'actual result: 6','ok':True}]}
    view=diagnostic_packet(report)
    assert view['status']=='uninterpreted_observations'
    assert not {'category','reason','next_steps'} & view.keys()
    assert view['diagnostic_observations']==report['diagnostic_observations']
    assert report['category']=='implementation'  # Audit record is not rewritten.


def test_repeated_failing_test_diagnosis_reaches_actor_and_still_requires_check(tmp_path):
    original = 'def double(x): return 2*x\n'
    (tmp_path/'numbers_impl.py').write_text(original)
    class Model(DiagnosisModel):
        step = 0
        async def __call__(self, messages, schema, effort):
            if schema.__name__ == 'FailureDiagnosis':
                return await super().__call__(messages, schema, effort)
            data = json.loads(messages[-1]['content'])
            if schema.__name__ == 'TaskContract':
                return schema.model_validate({'requirements': [
                    {'id': 'r1', 'text': 'Add regression tests', 'validation': 'new_or_updated_tests', 'source_quote': 'Add regression tests'},
                    {'id': 'r2', 'text': 'run checks', 'validation': 'registered_checks', 'source_quote': 'run checks'}]})
            if 'oneOf' not in schema.model_json_schema():
                return schema.model_validate({'aligned': True, 'ready': True, 'criteria': [
                    {'requirement': rid, 'status': 'supported', 'evidence': data['valid_evidence_ids'][-1:]}
                    for rid in ('r1', 'r2')], 'issues': [], 'next_mode': 'verify'})
            turns = [
                {'action': 'plan', 'plan': ['Inspect double, add the test, run and diagnose checks']},
                {'action': 'tool', 'tool': 'read', 'args': {'path': 'numbers_impl.py'}},
                {'action': 'tool', 'tool': 'write', 'args': {'path': 'test_double.py', 'content': 'from numbers_impl import double\nassert double(3)==5\n'}},
                {'action': 'tool', 'tool': 'check', 'args': {'name': 'unit'}},
                # An intervening edit used to reset the generic failure counter.
                {'action': 'tool', 'tool': 'edit', 'args': {'path': 'test_double.py', 'old': '==5', 'new': '==7'}},
                {'action': 'tool', 'tool': 'check', 'args': {'name': 'unit'}},
                {'action': 'tool', 'tool': 'edit', 'args': {'path': 'test_double.py', 'old': '==7', 'new': '==6'}},
                {'action': 'tool', 'tool': 'check', 'args': {'name': 'unit'}},
                {'action': 'finish', 'summary': 'Added the regression test and passed the registered check'},
            ]
            if self.step == 6:
                assert data['failed_checks'] == ['unit']
                assert data['cognitive_feedback'][-1]['failure_diagnosis']['category'] == 'test_expectation'
                assert not any(b['properties']['action']['const']=='finish' for b in schema.model_json_schema()['oneOf'])
            value = schema.model_validate(turns[self.step]); self.step += 1
            return value
    hitl.set_work_mode('auto')
    model = Model()
    harness = CodingHarness(tmp_path, model=model, config={'learn': False, 'max_turns': 15,
        'checks': {'unit': [sys.executable, '-B', 'test_double.py']},
        'orchestration': {'enabled': True, 'auto_domains': True, 'same_model': True,
                          'critic': {'enabled': True, 'active': False}}})
    result = asyncio.run(harness.run('Add regression tests for double(x), which returns twice x, and run checks'))
    assert result['status'] == 'done' and result['checks_passed'] == ['unit']
    assert (tmp_path/'numbers_impl.py').read_text() == original and model.calls == 1
