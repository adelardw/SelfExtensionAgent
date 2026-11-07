"""Context-preserving mode transitions, recursive scheduling and authority boundaries."""
import asyncio
import hashlib
import json
import sys

import pytest

from src.coding.control import SessionControl, Steered
from src.coding.decision import Branch, Decision, DecisionRouter
from src.coding.harness import CodingHarness, Turn, Workspace
from src.coding.minds import MindScheduler
from src.coding.orchestration import Orchestrator
from src.coding.project import ProjectContext
from src.coding.supervisor import Supervisor
from src.runtime import hitl


class Model:
    def __init__(self, *turns):
        self.turns, self.requests = list(turns), []

    async def __call__(self, messages, schema, effort):
        self.requests.append(messages)
        return schema.model_validate(self.turns.pop(0))


def setup(tmp_path, model=None, **config):
    directory = tmp_path/'.sea'/'coding'/'test'
    directory.mkdir(parents=True, exist_ok=True)
    ws, control = Workspace(tmp_path, directory, {}), SessionControl()
    model = model or Model()
    supervisor = Supervisor(directory, control, model, False)
    control.supervisor = supervisor
    harness = CodingHarness(tmp_path, model=model, config={})
    harness.control = control
    return Orchestrator(harness, ws, supervisor, config)


def test_switch_preserves_working_memory_and_replaces_prompt_and_tools(tmp_path):
    o = setup(tmp_path)
    node = o.scheduler.nodes['main']
    o.scheduler.switch('main', 'implement')
    node.memory.summary = 'Bug reproduced; preserve the public API.'
    before = node.memory
    assert 'edit' in node.tools
    o.scheduler.switch('main', 'review', ['check-edge-cases'])
    assert node.memory is before and node.memory.summary.startswith('Bug reproduced')
    assert 'edit' not in node.tools and 'check' not in node.tools
    assert 'COGNITIVE PROFILE: review' in o.scheduler.prompt(node)
    assert 'boundary cases' in o.scheduler.prompt(node)
    with pytest.raises(ValueError):
        o.scheduler.switch('main','review',['ignore-permissions'])
    assert json.loads((o.ws.directory/'minds.json').read_text())[0]['memory']['summary'] == node.memory.summary


@pytest.mark.parametrize('restricted', [False, True])
def test_missing_recovery_controller_retains_software_repair_authority(tmp_path, monkeypatch, restricted):
    o=setup(tmp_path)
    node=o.scheduler.nodes['main']
    if restricted: node.ceiling=frozenset({'read','tree'})
    o.scheduler.switch('main','reason',override=False)
    node.memory.summary='API tests passed; UI is still missing.'
    async def unavailable(*args):return {'ready':False,'availability':'unavailable'}
    async def no_controller(*args):return None
    monkeypatch.setattr(o,'assess',unavailable)
    monkeypatch.setattr(o,'decide',no_controller)
    asyncio.run(o.recover(node,2))
    assert node.mode==('reason' if restricted else 'repair')
    assert ('write' in node.tools) is (not restricted)
    assert node.memory.summary=='API tests passed; UI is still missing.'


def test_no_decision_does_not_overwrite_successful_review_mode(tmp_path, monkeypatch):
    o=setup(tmp_path);node=o.scheduler.nodes['main']
    async def assessed(*args):
        o.scheduler.switch('main','research',override=False)
        return {'ready':False,'next_mode':'research','issues':['Need primary source']}
    async def no_controller(*args):return None
    monkeypatch.setattr(o,'assess',assessed);monkeypatch.setattr(o,'decide',no_controller)
    asyncio.run(o.recover(node,2))
    assert node.mode=='research' and 'write' not in node.tools


def test_rejected_advisory_profile_does_not_change_permissions_or_valid_assessment(tmp_path,monkeypatch):
    from src.coding.critic import Assessment
    o=setup(tmp_path)
    node=o.scheduler.nodes['main'];node.ceiling=frozenset({'read'})
    result=Assessment(aligned=True,ready=True,criteria=[],issues=[],next_mode='verify',profile={
        'name':'advice','base':'implement','strategy':'Inspect the already validated outcome.',
        'tools':['write'],'exit_when':['done']})
    async def assess(*args):return result
    def reject(*args):raise ValueError('outside current capability ceiling')
    monkeypatch.setattr(o.critic,'assess',assess)
    monkeypatch.setattr(o.scheduler.profiles,'install',reject)
    report=asyncio.run(o.assess(node,'completion'))
    assert report['ready'] and not report['issues'] and node.ceiling==frozenset({'read'})
    assert node.mode=='inspect' and 'write' not in node.tools


def test_worker_can_correct_arguments_after_two_failed_calls(tmp_path):
    (tmp_path/'present.py').write_text('VALUE=2\n')
    class CorrectingModel(Model):
        async def __call__(self,messages,schema,effort):
            if len(self.requests)==2:
                packet=json.loads(messages[-1]['content'])
                assert packet['recovery']['blocked_calls']==[{'tool':'read','args':{'path':'missing.py'}}]
                assert 'read' in packet['capabilities']
            return await super().__call__(messages,schema,effort)
    failed={'action':'tool','tool':'read','args':{'path':'missing.py'}}
    model=CorrectingModel(failed,failed,
        {'action':'tool','tool':'read','args':{'path':'present.py'}},
        {'action':'finish','summary':'Read corrected path; VALUE is 2.'})
    o=setup(tmp_path,model,same_model=True,critic={'enabled':False})
    node=o.scheduler.create('main',Branch(mode='review',task='Inspect the value',tools=['read']))
    result=asyncio.run(o.run_mind(node))
    assert result=='Read corrected path; VALUE is 2.' and len(model.requests)==4
    assert 'VALUE=2' in node.memory.reads['present.py']


@pytest.mark.parametrize('context_policy',['bounded','single_action'])
def test_worker_result_memory_survives_observation_roll_and_stays_isolated(tmp_path,monkeypatch,context_policy):
    execute=Workspace.execute
    async def calculation(ws,name,args,**kwargs):
        if name=='math':return {'ok':True,'output':'{"ok":true,"result":"4","resolved":true}'}
        return await execute(ws,name,args,**kwargs)
    monkeypatch.setattr(Workspace,'execute',calculation)
    for i in range(4):(tmp_path/f'input{i}').write_text(str(i))
    class RetainingModel(Model):
        async def __call__(self,messages,schema,effort):
            if len(self.requests)==5:
                packet=json.loads(messages[-1]['content'])
                assert not any(x.get('tool')=='math' for x in packet['working_memory']['observations'])
                assert json.loads(packet['completed_results'][0]['output'])['result']=='4'
            return await super().__call__(messages,schema,effort)
    turns=[{'action':'tool','tool':'math','args':{'op':'simplify','expr':'2+2'}}]+[
        {'action':'tool','tool':'read','args':{'path':f'input{i}'}} for i in range(4)]+[
        {'action':'finish','summary':'Computed 4 and inspected the inputs.'}]
    o=setup(tmp_path,RetainingModel(*turns),same_model=True,critic={'enabled':False})
    o.harness.config['result_memory']=True
    o.harness.config['context_policy']=context_policy
    node=o.scheduler.create('main',Branch(mode='reason',task='Compute and inspect inputs',tools=['math','read']))
    result=asyncio.run(o.run_mind(node))
    assert result=='Computed 4 and inspected the inputs.'
    from src.coding.result_memory import ResultMemory
    assert not ResultMemory(o.ws,'main').packet(0)
    assert ResultMemory(o.ws,node.id).packet(0)


def test_worker_native_dialogue_records_its_own_actual_evidence(tmp_path):
    from src.coding.dialogue import active_dialogue
    (tmp_path/'worker-input').write_text('observed value')
    class RecordingModel(Model):
        async def __call__(self,messages,schema,effort):
            dialogue=active_dialogue.get()
            assert dialogue is not None
            if self.requests:
                assert len(dialogue.history)==1
                response=json.loads(dialogue.history[0][1]['content'])
                assert 'observed value' in response['output'] and response['evidence_id'].isdigit()
            else:
                dialogue.capture({'role':'assistant','content':None,'tool_calls':[{
                    'id':'child-read','type':'function','function':{
                        'name':'read','arguments':'{"path":"worker-input"}'}}]})
            return await super().__call__(messages,schema,effort)
    o=setup(tmp_path,RecordingModel({'action':'tool','tool':'read','args':{'path':'worker-input'}},
        {'action':'finish','summary':'Observed value'}),same_model=True,critic={'enabled':False})
    o.harness.config['native_history']=True
    node=o.scheduler.create('main',Branch(mode='review',task='Inspect worker input',tools=['read']))
    assert asyncio.run(o.run_mind(node))=='Observed value'


def test_subtask_ledger_retains_child_progress_and_marks_changed_parent_sources(tmp_path):
    o=setup(tmp_path,max_nodes=12)
    (tmp_path/'ports.py').write_text('original')
    child=o.scheduler.create('main',Branch(mode='review',task='Review input contract',tools=['read']))
    child.status='partial';child.memory.summary='Found missing type validation; review is not finished.'
    child.memory.files={'ports.py':o.ws.file_digest('ports.py')}
    child.memory.evidence=[o.ws.record({'ok':True,'output':'original'})]
    # No rolling tool observations are needed to retain this progress.
    row=o.subtask_ledger()['items'][0]
    assert row['id']==child.id and row['status']=='partial' and row['summary']==child.memory.summary
    assert row['summary_status']=='model_report_unverified' and row['evidence_present']
    assert not row['stale_parent_files']
    (tmp_path/'ports.py').write_text('changed')
    assert o.subtask_ledger()['items'][0]['stale_parent_files']==['ports.py']
    assert o.subtask_ledger(child.id)['items']==[]


def test_uncertain_recovery_does_not_interrupt_or_rebind_the_current_profile(tmp_path):
    o=setup(tmp_path)
    node=o.scheduler.nodes['main'];node.selection=frozenset({'read'})
    generation=node.generation
    async def assess(*args): return None
    async def decide(*args):
        return Decision(operation='continue',mode='inspect',branches=[],rules=[],reason='Uncertain')
    o.assess,o.decide=assess,decide
    asyncio.run(o.recover(node,2))
    assert node.generation==generation and node.selection==frozenset({'read'})


def test_diagnostic_feedback_expires_on_edit_external_change_or_steering(tmp_path):
    o=setup(tmp_path)
    node=o.scheduler.nodes['main'];ws=o.ws
    (tmp_path/'module.py').write_text('VALUE=1')
    feedback=[{'failure_diagnosis':{'category':'implementation','scope':{
        'revision':ws.revision,'steering_version':o.control.version(node.id),
        'source_hashes':{'module.py':ws.file_digest('module.py')}}}}]
    assert o.current_feedback(node,feedback)==feedback
    ws.revision+=1
    assert not o.current_feedback(node,feedback)
    ws.revision-=1
    (tmp_path/'module.py').write_text('VALUE=2')
    assert not o.current_feedback(node,feedback)
    (tmp_path/'module.py').write_text('VALUE=1')
    o.control.post('New task constraint',interrupt=False)
    assert not o.current_feedback(node,feedback)


def test_readonly_assignment_cannot_become_writer_even_with_isolated_writes(tmp_path):
    o=setup(tmp_path,isolated_writes=True)
    parent=o.scheduler.nodes['main']
    (tmp_path/'app.py').write_text('VALUE=1')
    parent.memory.reads={'app.py':'VALUE=1'}
    child=o.scheduler.create('main',Branch(mode='review',task='Read-only review',tools=[]))
    assert child.memory.reads==parent.memory.reads and child.memory.reads is not parent.memory.reads
    o.scheduler.switch(child.id,'implement')
    assert not child.tools & {'edit','write','run','accept'}
    nested=o.scheduler.create(child.id,Branch(mode='repair',task='Try to change code',tools=[]))
    assert not nested.ceiling & {'edit','write','run','accept'}


def test_root_source_text_handoff_and_nested_delegation_require_real_work(tmp_path):
    o=setup(tmp_path,isolated_writes=True)
    o.update_main('Fix module','implement','source inspected',['fix'],[],{'app.py':'hash'},
                  {'app.py':'exact source text'})
    child=o.scheduler.create('main',Branch(mode='review',task='Review module',tools=[]))
    assert child.memory.reads=={'app.py':'exact source text'}
    packet=child.memory.packet(o.ws);packet['handoff'].pop('goal',None)
    assert 'goal' in child.memory.inherited
    assert not o.delegation_ready(child)
    child.memory.plan=['Inspect independent scopes']
    o.supervisor.action(child.id,'read',True,'1','a')
    assert not o.delegation_ready(child)
    o.supervisor.action(child.id,'search',True,'2','b')
    assert o.delegation_ready(child)


def test_root_delegation_requires_actual_evidence_and_runtime_gate_does_not_buy_a_decision(tmp_path):
    o=setup(tmp_path)
    root=o.scheduler.nodes['main']
    assert not o.delegation_ready(root)
    root.memory.plan=['Inspect the unknown before splitting']
    o.supervisor.action('main','think',True,'1','mode-only')
    assert not o.delegation_ready(root)
    result=asyncio.run(o.delegate('Copy the full goal'))
    assert result[0]['status']=='not_ready' and o.router.calls==0 and o.scheduler.created==0
    o.supervisor.action('main','math',False,'2','concrete failed calculation')
    assert o.delegation_ready(root)


def test_exhausted_actions_are_removed_from_schema_and_explained_in_prompt():
    from src.coding.orchestration import scoped_schema
    from src.coding.prompts import action_policy
    schema=scoped_schema(Turn,{'read','edit'},can_delegate=False,can_plan=False)
    actions={b['properties']['action']['const'] for b in schema.model_json_schema()['oneOf']}
    assert actions=={'tool','finish'}
    assert 'No new delegation' in action_policy(schema) and 'execute it' in action_policy(schema)


def test_fork_inherits_compact_parent_state_not_private_sibling_context(tmp_path):
    o = setup(tmp_path)
    root = o.update_main('Implement a parser', 'implement', 'Delimiter is comma', ['preserve API'], [], {})
    eid = o.ws.record({'ok':True,'output':'delimiter inspection'})
    root.memory.evidence = [eid]
    a = o.scheduler.create('main', Branch(mode='inspect',task='Check parser',tools=[]))
    a.memory.summary = 'Sibling-only finding'
    b = o.scheduler.create('main', Branch(mode='review',task='Check types',tools=[]))
    assert b.memory.inherited['parent_summary'] == 'Delimiter is comma'
    assert b.memory.evidence == [eid]
    assert 'Sibling-only' not in str(b.memory.packet(o.ws))
    assert 'edit' not in a.tools
    c = o.scheduler.create(a.id, Branch(mode='implement',task='Narrow types',tools=['read','edit','run']))
    assert c.tools == {'read'}  # Switching a child to implement never grants write authority.


def test_handoff_evidence_validation_detects_missing_and_stale_sources(tmp_path):
    o = setup(tmp_path)
    f = tmp_path/'api.py'; f.write_text('old')
    memory = o.scheduler.nodes['main'].memory
    memory.files['api.py'] = hashlib.sha256(f.read_bytes()).hexdigest()
    memory.evidence = ['999']
    memory.reads = {'api.py':'old source'}
    f.write_text('new')
    packet = memory.packet(o.ws)
    assert packet['validation']['missing_evidence'] == ['999']
    assert packet['validation']['stale_files'] == ['api.py']
    assert packet['inspected_content'] == {}
    assert not packet['validation']['semantic_truth_verified']


def test_switch_during_child_generation_discards_old_result_and_keeps_context(tmp_path):
    async def scenario():
        started = asyncio.Event()
        class SwitchingModel(Model):
            async def __call__(self, messages, schema, effort):
                self.requests.append(messages)
                if len(self.requests) == 1:
                    return Turn(action='tool',tool='read',args={'path':'api.py'},summary='Looking for the boundary')
                if len(self.requests) == 2:
                    started.set()
                    try: await asyncio.sleep(30)
                    except asyncio.CancelledError:
                        return Turn(action='tool',tool='write',args={'path':'bad','content':'stale'})
                context = json.loads(messages[-1]['content'])
                assert context['phase'] == 'review'
                assert context['working_memory']['summary'] == 'Looking for the boundary'
                assert 'api.py' in context['working_memory']['inspected_content']
                assert 'evidence_ids' in context['working_memory']
                assert 'COGNITIVE PROFILE: review' in messages[0]['content']
                return Turn(action='finish',summary='Boundary reviewed; evidence 1')
        o = setup(tmp_path, SwitchingModel())
        (tmp_path/'api.py').write_text('limit = 10')
        node = o.scheduler.create('main',Branch(mode='inspect',task='Inspect boundary',tools=[]))
        task = asyncio.create_task(o.run_mind(node))
        await started.wait()
        o.scheduler.switch(node.id,'review')
        result = await asyncio.wait_for(task,2)
        assert result.startswith('Boundary reviewed')
        assert not (tmp_path/'bad').exists()
        assert len(o.model.requests) == 3
    asyncio.run(scenario())


def test_parallel_children_use_one_slot_and_nested_work_cannot_deadlock(tmp_path):
    async def scenario():
        o = setup(tmp_path, max_nodes=8, max_depth=3, max_branches=3, local_slots=1)
        active = peak = 0
        async def generate():
            nonlocal active, peak
            active += 1; peak = max(peak,active)
            await asyncio.sleep(.001)
            active -= 1
            return 'fact'
        async def runner(node):
            await o.scheduler.call(node, generate)
            if node.depth < 2:
                children = [Branch(mode='inspect',task='nested',tools=[])]
                await o.scheduler.group(node.id,children,runner)
            return 'done'
        branches = [Branch(mode='inspect',task=str(i),tools=[]) for i in range(50)]
        result = await asyncio.wait_for(o.scheduler.group('main',branches,runner),2)
        assert len(result) == 3 and len(o.scheduler.nodes) == 7 and peak == 1
        await o.scheduler.group('main',branches,runner)
        assert len(o.scheduler.nodes) == 8
    asyncio.run(scenario())


def test_target_interrupt_does_not_cancel_unrelated_concurrent_generation(tmp_path):
    async def scenario():
        o = setup(tmp_path,local_slots=2)
        a,b = [o.scheduler.create('main',Branch(mode='inspect',task=x,tools=[])) for x in ['a','b']]
        started = [asyncio.Event(),asyncio.Event()]
        finish = asyncio.Event()
        async def work(i):
            started[i].set(); await finish.wait(); return i
        ta = asyncio.create_task(o.scheduler.call(a,lambda:work(0)))
        tb = asyncio.create_task(o.scheduler.call(b,lambda:work(1)))
        await asyncio.gather(*(e.wait() for e in started))
        o.scheduler.switch(a.id,'reason')
        finish.set()
        results = await asyncio.gather(ta,tb,return_exceptions=True)
        assert isinstance(results[0],Steered) and results[1] == 1
    asyncio.run(scenario())


def test_router_cache_and_call_budget_and_strict_schema():
    async def scenario():
        model = Model({'operation':'select','mode':'review','branches':[],'reason':'edge cases','rules':['check-edge-cases']})
        router = DecisionRouter(model,{'max_calls':1})
        a = await router.choose('review',{'allowed_tools':['read']})
        assert await router.choose('review',{'allowed_tools':['read']}) is a
        assert await router.choose('other',{}) is None
        assert len(model.requests) == 1
        assert Decision.model_json_schema()['additionalProperties'] is False
        assert Decision.model_json_schema()['$defs']['Branch']['additionalProperties'] is False
    asyncio.run(scenario())


def test_live_switch_api_preserves_memory_and_rejects_invalid_target(tmp_path,monkeypatch):
    import src.interface.server as server
    from fastapi import HTTPException
    o = setup(tmp_path)
    o.control.scheduler = o.scheduler
    o.scheduler.nodes['main'].memory.summary = 'Keep this fact'
    monkeypatch.setattr(server,'_RUNS',{'run':{'status':'running','control':o.control}})
    result = asyncio.run(server.steer_run('run',server.SteerIn(action='switch',mode='review')))
    assert result['context_preserved'] and o.scheduler.nodes['main'].mode == 'review'
    assert server.coding_context('run')['cognition'][0]['memory']['summary'] == 'Keep this fact'
    with pytest.raises(HTTPException):
        asyncio.run(server.steer_run('run',server.SteerIn(action='switch',mode='repair',target='missing')))


def test_root_automatic_profiles_follow_plan_edit_check(tmp_path,monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'auto'})
    model = Model({'action':'plan','plan':['write and test']},
                  {'action':'tool','tool':'write','args':{'path':'result.py','content':'VALUE=2'}},
                  {'action':'tool','tool':'check','args':{'name':'unit'}},
                  {'action':'finish','summary':'Verified'})
    config = {'orchestration':{'enabled':True},'checks':{'unit':[sys.executable,'-c','import result; assert result.VALUE==2']}}
    result = asyncio.run(CodingHarness(tmp_path,model=model,config=config).run('Create result'))
    assert result['status'] == 'done'
    profiles = [json.loads(r[-1]['content'])['cognitive_mode'] for r in model.requests]
    # A write may be one file of a larger change; only the explicit successful
    # check moves into verification. The completion gate still enforces it.
    assert profiles == ['plan','implement','implement','verify']
    assert result['decision_usage']['calls'] == 0  # Normal work never calls a router.


def test_project_notes_require_human_preference_evidence_and_expire(tmp_path):
    o = setup(tmp_path)
    notes = ProjectContext(o.ws,o.control)
    with pytest.raises(ValueError):
        notes.add('preference','Style','Inferred personality')
    receipt = o.control.post('Prefer concise Russian replies')
    n = notes.add('preference','Reply style','Use concise Russian',[receipt['id']])
    assert n['status'] == 'hypothesis'
    assert (tmp_path/'.sea'/'context'/'INDEX.md').exists()
    (tmp_path/'api.py').write_text('v1')
    eid = o.ws.record({'ok':True,'output':'v1'})
    n = notes.add('fact','API version','v1',[eid],['api.py'])
    (tmp_path/'api.py').write_text('v2')
    assert not notes.valid(n)


def test_recovery_with_existing_plan_can_return_to_writing(tmp_path):
    o = setup(tmp_path)
    node = o.update_main('fix','implement','reproduced',['fix and check'],[],{})
    assert 'edit' in node.tools
    node = o.update_main('fix','plan','two failures',['fix and check'],[],{})
    assert node.mode == 'repair' and 'edit' in node.tools
    o.scheduler.switch('main','review')
    o.update_main('fix','implement','operator override',['fix'],[],{})
    assert node.mode == 'review' and 'edit' not in node.tools


def test_all_operator_constraints_survive_and_parent_rules_reach_descendants(tmp_path):
    o = setup(tmp_path)
    a = o.scheduler.create('main',Branch(mode='inspect',task='parent',tools=[]))
    b = o.scheduler.create(a.id,Branch(mode='inspect',task='nested',tools=[]))
    for i in range(16): o.control.post('constraint '+str(i),interrupt=False)
    o.control.post('Only inspect the API',target=a.id,interrupt=False)
    sibling = o.scheduler.create('main',Branch(mode='inspect',task='sibling',tools=[]))
    assert len(o.control.instructions()) == 16
    assert len(o.control.instructions(b.id)) == 17
    assert len(o.control.instructions(sibling.id)) == 16


def test_unknown_check_name_does_not_poison_completion(tmp_path,monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'auto'})
    model=Model({'action':'plan','plan':['write then test']},
                {'action':'tool','tool':'write','args':{'path':'result.py','content':'VALUE=2'}},
                {'action':'tool','tool':'check','args':{'name':'test_result.py'}},
                {'action':'tool','tool':'check','args':{'name':'unit'}},
                {'action':'finish','summary':'Verified'})
    config={'checks':{'unit':[sys.executable,'-c','import result; assert result.VALUE==2']}}
    result=asyncio.run(CodingHarness(tmp_path,model=model,config=config).run('create'))
    assert result['status']=='done' and result['checks_passed']==['unit']
