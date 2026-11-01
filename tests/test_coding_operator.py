"""Steering races, observable context, scoped sharing and validated learning."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from src.coding.control import SessionControl, Steered
from src.coding.context import ContextBoard, ExperienceStore
from src.coding.harness import CodingHarness, Turn, Workspace
from src.coding.request import fit_messages
from src.coding.model import LlamaCppModel
from src.runtime import hitl


@pytest.fixture(autouse=True)
def auto(monkeypatch):
    monkeypatch.setattr(hitl, '_work_mode', {'': 'auto'})


class Model:
    def __init__(self, *turns): self.turns = list(turns); self.requests = []
    async def __call__(self, messages, schema, effort):
        self.requests.append(messages)
        value = self.turns.pop(0)
        return schema.model_validate(value)


def test_steering_cancels_generation_and_stale_tool_never_executes(tmp_path):
    async def scenario():
        started = asyncio.Event()
        class Slow(Model):
            async def __call__(self, messages, schema, effort):
                self.requests.append(messages)
                if len(self.requests) == 1:
                    return Turn(action='plan', plan=['write a file'])
                if len(self.requests) == 2:
                    started.set()
                    try: await asyncio.sleep(30)
                    except asyncio.CancelledError:
                        return Turn(action='tool', tool='write', args={'path':'forbidden','content':'bad'})
                assert 'Do not modify files' in str(messages)
                return Turn(action='finish', summary='Inspection only')
        model, control = Slow(), SessionControl()
        task = asyncio.create_task(CodingHarness(tmp_path, model=model, config={}).run('write', control=control))
        await started.wait()
        control.post('Do not modify files; inspect only')
        result = await asyncio.wait_for(task, 2)
        assert result['status'] == 'done' and not (tmp_path/'forbidden').exists()
        assert len(model.requests) == 3
    asyncio.run(scenario())


def test_no_interrupt_means_next_boundary_not_lost_message():
    async def scenario():
        control = SessionControl()
        started, finish = asyncio.Event(), asyncio.Event()
        async def generation(): started.set(); await finish.wait(); return 'old response'
        task = asyncio.create_task(control.call(generation))
        await started.wait()
        receipt = control.post('new constraint', interrupt=False)
        assert not receipt['interrupted_generation']
        finish.set()
        assert await task == 'old response'
        assert control.drain()[0]['text'] == 'new constraint'
    asyncio.run(scenario())


def test_scoped_inboxes_and_prestart_message_persistence(tmp_path):
    control = SessionControl()
    control.post('child instruction', target='worker-1')
    control.post('parent instruction')
    control.bind(tmp_path/'steering.jsonl')
    assert [e['text'] for e in control.drain()] == ['parent instruction']
    assert [e['text'] for e in control.drain('worker-1')] == ['child instruction']
    restored = SessionControl(); restored.bind(tmp_path/'steering.jsonl', resume=True)
    assert len(restored.history) == 2


def test_board_requires_evidence_and_invalidates_changed_file(tmp_path):
    directory = tmp_path/'.sea'/'coding'; directory.mkdir(parents=True)
    ws = Workspace(tmp_path, directory)
    (tmp_path/'api.py').write_text('VERSION=1')
    eid = ws.record({'ok':True, 'output':'API inspection'})
    item = ws.board.publish('API version one', [eid], ['api.py'], trigger='API')
    assert ws.board.retrieve('API')[0]['id'] == item['id']
    (tmp_path/'api.py').write_text('VERSION=2')
    assert ws.board.retrieve('API') == []
    assert ws.board.inventory()[0]['valid'] is False
    with pytest.raises(ValueError): ws.board.publish('invented evidence', ['999'])


def test_experience_is_historical_verified_and_scoped_to_stack(tmp_path):
    (tmp_path/'pyproject.toml').write_text('version = "1"')
    store = ExperienceStore(tmp_path)
    assert store.promote('API validation', 'check types', [], [], []) is None
    entry = store.promote('API validation', 'check types before conversion', [{'tool':'check'}], ['unit'], ['1'])
    assert store.retrieve('API')[0]['id'] == entry['id']
    (tmp_path/'pyproject.toml').write_text('version = "2"')
    assert ExperienceStore(tmp_path).retrieve('API') == []
    assert ExperienceStore(tmp_path.parent).retrieve('API') == []


def test_request_receipt_matches_retained_entries_and_json_stays_valid():
    payload = {'task':'inspect', 'phase':'plan', 'human_steering':[{'id':'u1','text':'Keep API stable'}],
               'experience':[{'id':'old','text':'a'*9000},{'id':'new','text':'b'*9000}],
               'shared_context':[], 'recent':[{'output':'x'*10000}]*4}
    messages, receipt = fit_messages([{'role':'system','content':'policy'}, {'role':'user','content':json.dumps(payload)}], Turn, 16384, 4096)
    actual = json.loads(messages[-1]['content'])
    assert receipt['experience'] == [x['id'] for x in actual['experience']]
    assert receipt['steering'] == ['u1']
    assert actual['human_steering'][0]['text'] == 'Keep API stable'
    assert len(messages[-1]['content']) < 12000


def test_required_human_constraints_are_not_silently_clipped():
    data = {'task':'fix','human_steering':[{'id':'u','text':'constraint '*3000}]}
    with pytest.raises(ValueError, match='steering'):
        fit_messages([{'role':'system','content':'policy'},{'role':'user','content':json.dumps(data)}], Turn, 16384, 4096)


def test_live_stream_is_observable_before_completion(monkeypatch):
    events = []
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for payload in [
                {'choices':[{'delta':{'reasoning_content':'inspect inputs'}}]},
                {'choices':[{'delta':{'content':'{"action":"finish","summary":"done"}'}}]},
                {'choices':[], 'usage':{'prompt_tokens':10,'completion_tokens':8}},
            ]:
                yield ('data: '+json.dumps(payload)+'\n\n').encode()
            yield b'data: [DONE]\n\n'
    def transport(req):
        if req.url.path == '/v1/models': return httpx.Response(200,json={'data':[{'id':'sea-local','meta':{'n_params':4e9}}]})
        if req.url.path == '/props': return httpx.Response(200,json={})
        return httpx.Response(200, headers={'content-type':'text/event-stream'}, stream=Stream())
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(transport)))
    model = LlamaCppModel({}); model.on_event=lambda kind,data:events.append((kind,data))
    turn = asyncio.run(model([{'role':'user','content':'inspect'}],Turn,'medium'))
    assert turn.action == 'finish'
    assert [e[0] for e in events] == ['request','reasoning','content','response','validated','end']
    assert model.usage()['tokens'] == 18 and model.usage()['reasoning_calls'] == 1


def test_readonly_developer_commands_dont_invalidate_checks(tmp_path):
    import subprocess
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'app.py').write_text('first\n')
    subprocess.run(['git','-C',str(tmp_path),'add','app.py'],check=True)
    (tmp_path/'app.py').write_text('second\n')
    directory = tmp_path/'.sea'/'console'; directory.mkdir(parents=True)
    ws = Workspace(tmp_path,directory)
    diff = asyncio.run(ws.execute('diff',{},readonly=True))
    assert diff['ok'] and '+second' in diff['output']
    grep = asyncio.run(ws.execute('grep',{'pattern':'second'},readonly=True))
    assert 'app.py:1' in grep['output']
    assert ws.revision == 0


def test_candidate_learning_needs_successful_run(tmp_path):
    directory=tmp_path/'.sea'/'console'; directory.mkdir(parents=True)
    ws=Workspace(tmp_path,directory)
    eid=ws.record({'ok':True,'output':'official technique'})
    result=asyncio.run(ws.execute('learn',{'trigger':'parse inputs','lesson':'validate before cast','evidence':[eid]}))
    assert result['ok'] and ws.candidates
    assert not ExperienceStore(tmp_path).items


def test_supervisor_observation_has_no_model_calls(tmp_path):
    from src.coding.supervisor import Supervisor
    control, model = SessionControl(), Model()
    supervisor = Supervisor(tmp_path,control,model,enabled=False)
    supervisor.register('worker-1','review','inspect API')
    with supervisor.scope('worker-1'):
        supervisor.model_event('request',{'messages':[{'role':'user','content':'context'}],'receipt':{'shared':['ctx1']}})
        supervisor.model_event('reasoning','Consider the API contract')
    supervisor.action('worker-1','read',False,'1')
    assert asyncio.run(supervisor.review('task',[],anomaly='failure')) is None
    view=supervisor.view(include_context=True)
    assert view['agents'][1]['reasoning'] == 'Consider the API contract'
    assert view['agents'][1]['context'][0]['content'] == 'context'
    assert not model.requests
