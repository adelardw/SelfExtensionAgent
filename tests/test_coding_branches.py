"""Real isolated editing, conflict detection and parent patch/check lifecycle."""
import asyncio
import json
import sys

import pytest

from src.coding.branches import BranchStore
from src.coding.control import SessionControl
from src.coding.harness import Workspace, CodingHarness
from src.coding.profiles import ProfileRegistry
from src.runtime import hitl


@pytest.fixture(autouse=True)
def auto(monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'auto'})


def setup(tmp_path, checks=None):
    directory=tmp_path/'.sea'/'coding'/'test';directory.mkdir(parents=True)
    ws=Workspace(tmp_path,directory,checks or {'unit':[sys.executable,'-c','from app import value; assert value()==2']})
    (tmp_path/'app.py').write_text('def value():\n    return 1\n')
    control=SessionControl();control.execution='auto'
    store=BranchStore(ws,control,{})
    ws.proposal_store,ws.owner_id=store,'main'
    return ws,control,store


def test_snapshot_check_keeps_virtualenv_and_installed_dependencies(tmp_path):
    # Resolving bin/python's symlink silently drops pyvenv.cfg and site-packages.
    # A stdlib-only check cannot catch this; pytest must import in the snapshot.
    code = ('import sys, pytest; '
            'assert sys.prefix == '+repr(sys.prefix)+'; '
            'from app import value; assert value() == 1; print("environment preserved")')
    ws, control, store = setup(tmp_path, {'unit':[sys.executable,'-c',code]})
    try:
        result=asyncio.run(ws.snapshot_check('unit'))
        assert result['ok'], result['output']
        assert 'environment preserved' in result['output']
    finally:
        store.close()


def test_readonly_diagnostic_does_not_make_current_check_or_failure_stale(tmp_path):
    from src.coding.progress import FailureFocus
    async def run():
        ws, control, store=setup(tmp_path,{'unit':[sys.executable,'-c','from app import value; assert value()==1']})
        try:
            assert (await ws.execute('check',{'name':'unit'}))['ok']
            probe={'argv':[sys.executable,'-c','import sys; print(sys.prefix)']}
            assert (await ws.execute('run',probe))['ok']
            assert ws.passed=={'unit'} and ws.verified_revision==ws.revision
            await ws.execute('edit',{'path':'app.py','old':'return 1','new':'return 2'})
            failed=await ws.execute('check',{'name':'unit'})
            assert not failed['ok']
            focus=FailureFocus(ws)
            focus.observe(ws,'check',{'name':'unit'},failed,ws.record(failed),0,['app.py'])
            assert (await ws.execute('run',probe))['ok']
            assert not focus.packet(ws,0)['source_changed_since_failure']
            child=store.fork(ws,'mind-1','main')
            revision=child.revision
            changed=await child.execute('run',{'argv':[sys.executable,'-c',
                'from pathlib import Path; Path("app.py").write_text("def value(): return 3\\n")']})
            assert changed['ok'] and child.revision>revision
            assert (child.root/'app.py').read_text().endswith('return 3\n')
        finally: store.close()
    asyncio.run(run())


def test_child_edits_only_snapshot_then_parent_adopts_checked_patch(tmp_path):
    async def run():
        ws,control,store=setup(tmp_path)
        child=store.fork(ws,'mind-1','main')
        result=await child.execute('edit',{'path':'app.py','old':'return 1','new':'return 2'})
        assert result['ok'] and 'return 1' in (tmp_path/'app.py').read_text()
        check=await child.execute('check',{'name':'unit'})
        assert check['ok'],check
        proposal=store.propose('mind-1')
        assert proposal['files']==['app.py']
        with pytest.raises(ValueError,match='Inspect patch'):
            await store.accept(ws,'main','mind-1')
        assert '+    return 2' in store.patch('main','mind-1')
        result=await store.accept(ws,'main','mind-1')
        assert result['ok'],result
        assert 'return 2' in (tmp_path/'app.py').read_text()
        assert not ws.passed and ws.verified_revision==-1  # Staging checks do not forge parent verification.
        assert (await ws.execute('check',{'name':'unit'}))['ok']
        assert store.proposals['mind-1']['status']=='accepted'
        store.close()
    asyncio.run(run())


def test_parent_dirty_changes_are_preserved_and_same_file_conflict_rejected(tmp_path):
    async def run():
        ws,control,store=setup(tmp_path)
        child=store.fork(ws,'mind-1','main')
        await child.execute('edit',{'path':'app.py','old':'return 1','new':'return 2'})
        store.propose('mind-1');store.patch('main','mind-1')
        (tmp_path/'app.py').write_text('user edit\n')
        result=await store.accept(ws,'main','mind-1')
        assert not result['ok'] and result['conflicts']==['app.py']
        assert (tmp_path/'app.py').read_text()=='user edit\n'
        store.close()
    asyncio.run(run())


def test_combined_check_failure_does_not_touch_parent(tmp_path):
    async def run():
        ws,control,store=setup(tmp_path)
        child=store.fork(ws,'mind-1','main')
        await child.execute('edit',{'path':'app.py','old':'return 1','new':'return 3'})
        store.propose('mind-1');store.patch('main','mind-1')
        result=await store.accept(ws,'main','mind-1')
        assert not result['ok'] and 'Combined branch check failed' in result['output']
        assert 'return 1' in (tmp_path/'app.py').read_text()
        assert store.proposals['mind-1']['status']=='proposed'
        store.close()
    asyncio.run(run())


def test_operator_message_during_validation_prevents_adoption(tmp_path):
    async def run():
        ws,control,store=setup(tmp_path)
        child=store.fork(ws,'mind-1','main')
        await child.execute('edit',{'path':'app.py','old':'return 1','new':'return 2'})
        store.propose('mind-1');store.patch('main','mind-1')
        real_fork=store.fork
        def fork(*args):
            staging=real_fork(*args)
            real_execute=staging.execute
            async def execute(*a,**kw):
                result=await real_execute(*a,**kw)
                control.post('Stop changing files',interrupt=False)
                return result
            staging.execute=execute
            return staging
        store.fork=fork
        result=await store.accept(ws,'main','mind-1')
        assert not result['ok'] and 'Operator changed' in result['output']
        assert 'return 1' in (tmp_path/'app.py').read_text()
        store.close()
    asyncio.run(run())


def test_custom_profile_is_generated_strategy_with_bounded_tools(tmp_path):
    profiles=ProfileRegistry(tmp_path)
    name=profiles.install({'name':'parser-boundary-audit','base':'review',
        'strategy':'Inspect numeric conversion boundaries and compare callers before proposing changes.',
        'tools':['read','edit','check'],'exit_when':['All parser callers and boundary cases inspected']},{'read'})
    assert profiles.tools(name)=={'read'}
    assert 'numeric conversion' in profiles.prompt(name)
    assert json.loads((tmp_path/'profiles.json').read_text())[-1]['custom']
    with pytest.raises(ValueError):
        profiles.install({'name':'review','base':'review','strategy':'overwrite built-in instructions',
                          'tools':[],'exit_when':['done']},{'read'})


def test_recursive_writing_worker_patch_is_adopted_by_real_harness(tmp_path):
    class Model:
        def __init__(self): self.main=0;self.child=0
        async def __call__(self,messages,schema,effort):
            packet=json.loads(messages[-1]['content'])
            if 'working_memory' in packet:
                self.child+=1
                values=[{'action':'plan','plan':['fix and check']},
                    {'action':'tool','tool':'read','args':{'path':'app.py'}},
                    {'action':'tool','tool':'edit','args':{'path':'app.py','old':'return 1','new':'return 2'}},
                    {'action':'tool','tool':'check','args':{'name':'unit'}},
                    {'action':'finish','summary':'Return value fixed and unit checked'}]
                return schema.model_validate(values[self.child-1])
            self.main+=1
            values=[{'action':'plan','plan':['Delegate fix','adopt','check']},
                {'action':'tool','tool':'read','args':{'path':'app.py'}},
                {'action':'delegate','role':'implement','task':'Fix app.py value to return 2 and check unit'},
                {'action':'tool','tool':'patch','args':{'id':'mind-1'}},
                {'action':'tool','tool':'accept','args':{'id':'mind-1'}},
                {'action':'tool','tool':'check','args':{'name':'unit'}},
                {'action':'finish','summary':'Implemented via isolated worker, adopted and checked'}]
            return schema.model_validate(values[self.main-1])
    (tmp_path/'app.py').write_text('def value():\n    return 1\n')
    config={'checks':{'unit':[sys.executable,'-c','from app import value; assert value()==2']},
            'orchestration':{'enabled':True,'isolated_writes':True,'mind_turns':10,'decision':{'max_calls':0}}}
    result=asyncio.run(CodingHarness(tmp_path,model=Model(),config=config).run('Fix value'))
    assert result['status']=='done' and result['checks_passed']==['unit']
    assert (tmp_path/'app.py').read_text().endswith('return 2\n')
    child=next(a for a in result['supervisor']['agents'] if a['id']=='mind-1')
    assert child['proposal']['files']==['app.py']


def test_dependent_patches_are_verified_and_applied_together(tmp_path):
    async def run():
        checks={'unit':[sys.executable,'-c','import a,b; assert a.VALUE+b.VALUE==4']}
        ws,control,store=setup(tmp_path,checks)
        (tmp_path/'a.py').write_text('VALUE=0\n');(tmp_path/'b.py').write_text('VALUE=0\n')
        for node,path in [('mind-1','a.py'),('mind-2','b.py')]:
            child=store.fork(ws,node,'main')
            await child.execute('edit',{'path':path,'old':'VALUE=0','new':'VALUE=2'})
            store.propose(node);store.patch('main',node)
        alone=await store.accept(ws,'main','mind-1')
        assert not alone['ok'] and (tmp_path/'a.py').read_text()=='VALUE=0\n'
        together=await store.accept_many(ws,'main',['mind-1','mind-2'])
        assert together['ok'],together
        assert all(store.proposals[n]['status']=='accepted' for n in ['mind-1','mind-2'])
        assert ws.revision==1
        store.close()
    asyncio.run(run())


def test_parent_checks_allow_temporary_test_outputs_without_writing_parent(tmp_path):
    async def run():
        ws,control,store=setup(tmp_path,{'unit':[sys.executable,'-c','from pathlib import Path; Path("test_output.csv").write_text("temporary"); assert Path("app.py").exists()']})
        result=await ws.execute('check',{'name':'unit'})
        assert result['ok'],result
        assert not (tmp_path/'test_output.csv').exists()
        assert result['check_workspace']=='isolated snapshot of current project'
    asyncio.run(run())


def test_readonly_child_think_switches_its_own_profile(tmp_path):
    from src.coding.orchestration import Orchestrator
    from src.coding.supervisor import Supervisor
    from src.coding.decision import Branch
    class Model: pass
    async def run():
        ws,control,store=setup(tmp_path)
        harness=CodingHarness(tmp_path,model=Model(),config={});harness.control=control
        supervisor=Supervisor(ws.directory,control,harness.model,False);control.supervisor=supervisor
        o=Orchestrator(harness,ws,supervisor,{'isolated_writes':True})
        child=o.scheduler.create('main',Branch(mode='inspect',task='inspect',tools=[]))
        readonly=o.workspace(child)
        assert readonly is not ws and readonly.root==ws.root
        result=await readonly.execute('think',{'profile':'review'},readonly=True)
        assert result['ok'] and child.mode=='review'
        assert o.scheduler.nodes['main'].mode=='inspect'
    asyncio.run(run())


def test_resume_restores_pending_patch_and_accepted_worker_provenance(tmp_path):
    async def run():
        ws,control,store=setup(tmp_path)
        child=store.fork(ws,'mind-1','main')
        await child.execute('edit',{'path':'app.py','old':'return 1','new':'return 2'})
        store.propose('mind-1');store.close()
        fresh=Workspace(tmp_path,ws.directory,ws.checks)
        restored=BranchStore(fresh,SessionControl(),{})
        nodes=restored.restore()
        assert nodes['mind-1'].path('app.py').read_text().endswith('return 2\n')
        assert restored.pending()[0]['id']=='mind-1'
        restored.patch('main','mind-1')
        assert (await restored.accept(fresh,'main','mind-1'))['ok']
        restored.close()
        again=BranchStore(fresh,SessionControl(),{});again.restore()
        assert again.proposals['mind-1']['status']=='accepted' and not again.pending()
        assert not again.snapshots['mind-1'].workspace.passed
        again.close()
    asyncio.run(run())


def test_observer_switches_stuck_worker_without_manual_intervention(tmp_path):
    from src.coding.orchestration import Orchestrator
    from src.coding.supervisor import Supervisor
    from src.coding.decision import Branch
    from src.coding.control import Steered
    class Model:
        async def __call__(self,messages,schema,effort):
            return schema.model_validate({'operation':'switch','mode':'reason','target':'mind-1',
                'branches':[],'rules':['compare-hypotheses'],'reason':'Repeated failed reads'})
    async def run():
        ws,control,store=setup(tmp_path)
        harness=CodingHarness(tmp_path,model=Model(),config={});harness.control=control
        supervisor=Supervisor(ws.directory,control,harness.model,False);control.supervisor=supervisor
        o=Orchestrator(harness,ws,supervisor,{'observer':True,'decision':{'max_calls':1}})
        child=o.scheduler.create('main',Branch(mode='inspect',task='Inspect API',tools=[]))
        child.memory.summary='Existing fact must survive'
        started=asyncio.Event()
        async def generation(): started.set();await asyncio.sleep(30)
        task=asyncio.create_task(o.scheduler.call(child,generation));await started.wait()
        o.start()
        supervisor.action(child.id,'read',False,'1','same-read')
        supervisor.action(child.id,'read',False,'2','same-read')
        result=await asyncio.wait_for(asyncio.gather(task,return_exceptions=True),2)
        assert isinstance(result[0],Steered)
        for _ in range(100):
            if child.mode=='reason': break
            await asyncio.sleep(.001)
        assert child.mode=='reason' and child.memory.summary=='Existing fact must survive'
        assert o.router.calls==1
        await o.close()
    asyncio.run(run())
