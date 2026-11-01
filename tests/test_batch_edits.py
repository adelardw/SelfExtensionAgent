"""Coherent edits retain single-file authority, partial-state truth and checks."""
import asyncio
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import ModuleType,SimpleNamespace

import pytest

from src.coding.harness import CodingHarness,Workspace,Turn
from src.coding.native_tools import compile_tools,decode_call
from src.coding.orchestration import scoped_schema
from src.runtime import hitl


@pytest.fixture(autouse=True)
def auto(monkeypatch):
    monkeypatch.setattr(hitl,'_work_mode',{'':'auto'})


def workspace(root):
    directory=root/'.sea';directory.mkdir()
    for path in ('impl.py','test_impl.py'):(root/path).write_text('VALUE=1\n')
    ws=Workspace(root,directory,{})
    ws.passed={'unit'};ws.verified_revision=0
    return ws


def replacement(path='impl.py',old='VALUE=1',new='VALUE=2'):
    return {'path':path,'old':old,'new':new}


def test_coherent_native_edit_keeps_capabilities_and_invalidates_checks(tmp_path):
    ws=workspace(tmp_path)
    schema=scoped_schema(Turn,{'edit','read'},False,False)
    definitions=compile_tools(schema)
    args={'changes':[replacement(),replacement('test_impl.py')]}
    call={0:{'name':'edit','arguments':json.dumps(args)}}
    turn=schema.model_validate(decode_call(call,definitions))
    result=asyncio.run(ws.execute(turn.tool,turn.args))
    assert result['ok'] and result['changed_paths']==['impl.py','test_impl.py']
    assert ws.revision==2 and not ws.passed and ws.verified_revision!=ws.revision
    assert ws.changed=={'impl.py','test_impl.py'}
    assert all((tmp_path/p).read_text()=='VALUE=2\n' for p in ws.changed)
    with pytest.raises(ValueError,match='capability'):
        decode_call(call,compile_tools(scoped_schema(Turn,{'read'},False,False)))


@pytest.mark.parametrize('kind',['readonly','plan','denied'])
def test_batch_does_not_bypass_execution_authority(tmp_path,monkeypatch,kind):
    ws=workspace(tmp_path)
    if kind=='plan':hitl.set_work_mode('plan')
    if kind=='denied':
        hitl.set_work_mode('manual')
        async def deny(*_):return False
        monkeypatch.setattr(hitl,'confirm',deny)
    result=asyncio.run(ws.execute('edit',{'changes':[replacement(),replacement('test_impl.py')]},readonly=kind=='readonly'))
    assert not result['ok'] and ws.revision==0 and not ws.changed
    assert (tmp_path/'impl.py').read_text()=='VALUE=1\n'


@pytest.mark.parametrize('bad',[{}, {'path':'x','old':'','new':'n'}, {'path':'x','old':1,'new':'n'}])
def test_all_replacement_shapes_validated_before_first_write(tmp_path,bad):
    ws=workspace(tmp_path)
    result=asyncio.run(ws.execute('edit',{'changes':[replacement(),bad]}))
    assert not result['ok'] and ws.revision==0 and not ws.changed


def test_partial_failure_reports_changes_and_never_runs_later_edits(tmp_path):
    ws=workspace(tmp_path)
    changes=[replacement(),replacement('.env'),replacement('test_impl.py')]
    (tmp_path/'.env').write_text('VALUE=1\n')
    result=asyncio.run(ws.execute('edit',{'changes':changes}))
    assert not result['ok'] and result['failed_index']==1
    assert result['completed_replacements']==1 and result['changed_paths']==['impl.py']
    assert ws.changed=={'impl.py'} and not ws.passed
    assert (tmp_path/'.env').read_text()==(tmp_path/'test_impl.py').read_text()=='VALUE=1\n'


def test_batch_bounds_and_mixed_forms_are_rejected(tmp_path):
    ws=workspace(tmp_path)
    for args in ({'changes':[replacement()]*9},{'changes':[replacement()],**replacement()},
                 {'changes':[replacement(new='я'*51000)]}):
        assert not asyncio.run(ws.execute('edit',args))['ok']
    assert ws.revision==0


def test_actor_receives_partial_batch_source_and_must_check_before_finish(tmp_path):
    (tmp_path/'impl.py').write_text('VALUE=1\n')
    class Model:
        step=0
        async def __call__(self,messages,schema,effort):
            packet=json.loads(messages[-1]['content'])
            if self.step==1:
                assert 'VALUE=2' in packet['inspected_content']['impl.py']
                assert packet['changed']==['impl.py'] and packet['verified'] is False
            turns=[{'action':'tool','tool':'edit','args':{'changes':[
                replacement(),replacement('missing.py')]}},
                {'action':'finish','summary':'premature'},
                {'action':'tool','tool':'check','args':{'name':'unit'}},
                {'action':'finish','summary':'Updated implementation and checked; second edit was rejected.'}]
            turn=turns[self.step];self.step+=1;return schema.model_validate(turn)
    h=CodingHarness(tmp_path,model=Model(),config={'planning':'adaptive','learn':False,
        'max_turns':4,'checks':{'unit':[sys.executable,'-B','-c','from impl import VALUE; assert VALUE==2']}})
    result=asyncio.run(h.run('Set VALUE to 2 and check.'))
    assert result['status']=='done' and result['checks_passed']==['unit']
    assert result['tools_used']==['edit','check']


def test_container_batch_uses_remote_files_and_permissions(tmp_path,monkeypatch):
    # Harbor is optional. Only its unrelated base class is stubbed; the actual
    # ContainerWorkspace and exact remote file helper execute unchanged.
    for name in ('harbor','harbor.agents','harbor.agents.base'):
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    sys.modules['harbor.agents.base'].BaseAgent=object
    source=Path(__file__).resolve().parents[1]/'src/eval/terminalbench_agent.py'
    spec=importlib.util.spec_from_file_location('batch_container_adapter',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    remote=tmp_path/'remote';remote.mkdir()
    host=tmp_path/'metadata';host.mkdir()
    for path in ('impl.py','test_impl.py'):(remote/path).write_text('VALUE=1\n')
    (host/'impl.py').write_text('host must remain untouched')
    class Environment:
        async def exec(self,command,timeout_sec):
            process=await asyncio.create_subprocess_exec(*shlex.split(command),
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            stdout,stderr=await asyncio.wait_for(process.communicate(),timeout_sec)
            return SimpleNamespace(return_code=process.returncode,stdout=stdout.decode(),stderr=stderr.decode())
    ws=module.ContainerWorkspace(host,host/'.sea',{},Environment());ws.remote_root=str(remote)
    args={'changes':[replacement(),replacement('test_impl.py')]}
    assert not asyncio.run(ws.execute('edit',args,readonly=True))['ok']
    result=asyncio.run(ws.execute('edit',args))
    assert result['ok'] and ws.changed=={'impl.py','test_impl.py'} and ws.revision==2
    assert (host/'impl.py').read_text()=='host must remain untouched'
    assert (remote/'impl.py').read_text()=='VALUE=2\n' and ws.hashes['test_impl.py']


def test_worker_batch_retains_both_sources_in_its_isolated_proposal(tmp_path):
    from src.coding.control import SessionControl
    from src.coding.decision import Branch
    from src.coding.orchestration import Orchestrator
    from src.coding.supervisor import Supervisor
    ws=workspace(tmp_path)
    ws.checks={'unit':[sys.executable,'-B','-c',
        'from impl import VALUE; from test_impl import VALUE as TEST; assert VALUE==TEST==2']}
    control=SessionControl();control.execution='auto'
    class Model:
        step=0
        async def __call__(self,messages,schema,effort):
            if self.step==1:
                memory=json.loads(messages[-1]['content'])['working_memory']
                assert all('VALUE=2' in memory['inspected_content'][p] for p in ('impl.py','test_impl.py'))
            turns=[{'action':'tool','tool':'edit','args':{'changes':[replacement(),replacement('test_impl.py')]}},
                   {'action':'tool','tool':'check','args':{'name':'unit'}},
                   {'action':'finish','summary':'Both files changed and checked in child.'}]
            turn=turns[self.step];self.step+=1;return schema.model_validate(turn)
    model=Model();h=CodingHarness(tmp_path,model=model,config={'planning':'adaptive','learn':False})
    h.control=control
    o=Orchestrator(h,ws,Supervisor(ws.directory,control,model,False),
        {'same_model':True,'isolated_writes':True,'mind_turns':4,'critic':{'enabled':False}})
    node=o.scheduler.create('main',Branch(mode='implement',task='Update both values to 2 and verify',tools=['edit','check']))
    try:
        result=asyncio.run(o.run_mind(node))
        assert sorted(result['proposal']['files'])==['impl.py','test_impl.py']
        assert (tmp_path/'impl.py').read_text()==(tmp_path/'test_impl.py').read_text()=='VALUE=1\n'
    finally:o.branches.close()
