"""Concurrent modes cooperate through evidence, without merging private contexts."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from src.coding.control import SessionControl
from src.coding.decision import Branch
from src.coding.harness import CodingHarness, Turn, Workspace
from src.coding.orchestration import Orchestrator
from src.coding.request import fit_messages
from src.coding.supervisor import Supervisor


def setup(tmp_path, model=None, backend='openrouter', **config):
    directory = tmp_path/'.sea'/'coding'/'parallel'
    directory.mkdir(parents=True)
    ws, control = Workspace(tmp_path, directory, {}), SessionControl()
    model = model or SimpleNamespace()
    supervisor = Supervisor(directory, control, model, False)
    control.supervisor = supervisor
    harness = CodingHarness(tmp_path, model=model, config={'backend':backend, 'navigation':{'backend':'lexical'}})
    harness.control = control
    return Orchestrator(harness, ws, supervisor, {'critic':{'enabled':False},
        'decision':{'max_calls':0}, 'local_slots':1, 'cloud_slots':2, **config})


@pytest.mark.parametrize('backend,expected',[('openrouter',2),('ollama',1),('local',1)])
def test_cloud_slots_do_not_increase_local_concurrency(tmp_path, backend, expected):
    o = setup(tmp_path, backend=backend)
    assert o.scheduler.slot_limit == expected


def test_parallel_modes_and_nested_children_release_generation_slots(tmp_path):
    async def scenario():
        o = setup(tmp_path, max_nodes=7, max_depth=3)
        active = peak = 0
        both_started = asyncio.Event()
        release = asyncio.Event()

        async def generation():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2: both_started.set()
            try:
                await release.wait()
            finally:
                active -= 1

        async def runner(node):
            await o.scheduler.call(node, generation)
            if node.depth == 1:
                await o.scheduler.group(node.id, [Branch(mode='review',task='nested '+node.id,tools=['read'])], runner)
            return 'completed own assignment'

        task = asyncio.create_task(o.scheduler.group('main', [
            Branch(mode=mode,task=mode,tools=[]) for mode in ('plan','research','implement')], runner))
        await asyncio.wait_for(both_started.wait(), 2)
        assert active == peak == 2 and not task.done()
        release.set()
        results = await asyncio.wait_for(task, 2)
        assert peak == 2 and len(o.scheduler.nodes) == 7
        assert all(row['status']=='done' for row in results)
        assert len([n for n in o.scheduler.nodes.values() if n.depth==2]) == 3
        assert not o.control._inflight
    asyncio.run(scenario())


def test_cancelling_group_finishes_children_and_retains_status(tmp_path):
    async def scenario():
        o = setup(tmp_path)
        started = asyncio.Event()

        async def runner(node):
            async def block():
                if len(o.control._inflight)==2: started.set()
                await asyncio.Event().wait()
            await o.scheduler.call(node, block)

        task = asyncio.create_task(o.scheduler.group('main', [
            Branch(mode=mode,task=mode,tools=[]) for mode in ('research','review')], runner))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert not o.control._inflight
        retained = json.loads((o.ws.directory/'minds.json').read_text())
        assert all(n['status']=='interrupted' for n in retained if n['id']!='main')
    asyncio.run(scenario())


def test_readonly_worker_shares_without_source_authority(tmp_path):
    async def scenario():
        o = setup(tmp_path)
        (tmp_path/'value.py').write_text('value = 7\n')
        node = o.scheduler.create('main', Branch(mode='research',task='find value',tools=[]))
        ws = o.workspace(node)
        eid = o.ws.record(await ws.execute('read',{'path':'value.py'}))
        assert 'share' in node.tools and not {'edit','write','run'} & node.tools
        result = await ws.execute('share', {'text':'Observed value 7','evidence':[eid], 'files':['value.py']})
        item = json.loads(result['output'])
        assert item['author'] == node.id and o.ws.board.valid(item)
        o.scheduler.switch(node.id, 'implement')
        assert not {'edit','write','run'} & node.tools
        assert node.memory is o.scheduler.nodes[node.id].memory
    asyncio.run(scenario())


def test_shared_finding_uses_publisher_snapshot_and_rejects_other_sessions(tmp_path):
    o = setup(tmp_path, isolated_writes=True)
    (tmp_path/'value.py').write_text('value = 1\n')
    node = o.scheduler.create('main', Branch(mode='implement',task='change value',tools=[]))
    ws = o.workspace(node)
    (ws.root/'value.py').write_text('value = 2\n')
    eid = o.ws.record({'ok':True,'output':'observed value = 2 in child'})
    item = ws.board.publish('Child has value 2',[eid],['value.py'],workspace=ws)
    assert item['author'] == node.id and ws.board.valid(item,ws)
    assert not ws.board.valid(item,o.ws)  # same path, different snapshot
    (tmp_path/'value.py').write_text('value = 2\n')
    assert ws.board.valid(item,o.ws)
    other_dir = tmp_path/'.sea'/'coding'/'other'
    other_dir.mkdir()
    other = Workspace(tmp_path,other_dir,{})
    with pytest.raises(ValueError,match='same session'):
        ws.board.publish('cross-session',[eid],workspace=other)


def test_concurrent_workers_see_finding_next_step_without_private_transcript(tmp_path):
    async def scenario():
        (tmp_path/'value.py').write_text('value = 7\n')
        started, published = asyncio.Event(), asyncio.Event()
        calls, packets = {}, []

        class Model:
            async def __call__(self, messages, schema, effort):
                packet = json.loads(messages[-1]['content'])
                name = packet['task']
                calls[name] = calls.get(name,0)+1
                n = calls[name]
                if n == 1:
                    if len(calls) == 2: started.set()
                    await started.wait()
                    return schema.model_validate({'action':'tool','tool':'read','args':{'path':'value.py'}})
                if name == 'publisher':
                    if n == 2:
                        return schema.model_validate({'action':'tool','tool':'share','args':{
                            'text':'Observed value 7','evidence':packet['working_memory']['evidence_ids'][-1:],
                            'files':['value.py'],'trigger':'unrelated search terms'}})
                    published.set()
                    return schema.model_validate({'action':'finish','summary':'Published observation.'})
                if n == 2:
                    await published.wait()
                    return schema.model_validate({'action':'tool','tool':'evidence','args':{
                        'id':packet['working_memory']['evidence_ids'][-1]}})
                packets.append(packet)
                return schema.model_validate({'action':'finish','summary':'Reviewed available evidence.'})

        o = setup(tmp_path, Model())
        branches = [Branch(mode=mode,task=task,tools=[]) for mode,task in
                    [('research','publisher'),('review','consumer')]]

        async def runner(node):
            if node.task=='publisher':
                node.memory.observations.append({'private_note':'PRIVATE_TRANSCRIPT_MARKER'})
            return await o.run_mind(node)

        results = await asyncio.wait_for(o.scheduler.group('main',branches,runner), 5)
        assert all(r['status']=='done' for r in results), results
        packet = packets[-1]
        assert 'PRIVATE_TRANSCRIPT_MARKER' not in json.dumps(packet)
        findings = packet['cooperation']['recent_findings']
        assert any(f['text']=='Observed value 7' for f in findings)
        peer = next(p for p in packet['cooperation']['peers'] if p['task']=='publisher')
        assert peer['mode']=='research' and peer['status']=='done'
        _, receipt = fit_messages([{'role':'system','content':'Act'},{'role':'user','content':json.dumps(packet)}],Turn,32768,1024)
        assert {f['id'] for f in findings} == set(receipt['peer_findings'])
        assert peer['id'] in receipt['peers']
        (tmp_path/'value.py').write_text('value = 8\n')
        consumer = next(n for n in o.scheduler.nodes.values() if n.task=='consumer')
        assert not o.cooperation(consumer,o.workspace(consumer))['recent_findings']
    asyncio.run(scenario())
