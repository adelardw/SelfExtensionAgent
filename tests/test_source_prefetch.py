"""Initial source retrieval must remain bounded, attributable and scope preserving."""
import asyncio
import copy
import hashlib
import json

import pytest

from src.coding.harness import CodingHarness,Workspace
from src.coding.source_prefetch import prefetch_sources


def workspace(root):
    directory=root/'.sea';directory.mkdir()
    return Workspace(root,directory,{})


def prepare(ws,task='Inspect module.py',budget=4800,**kwargs):
    return asyncio.run(prefetch_sources(ws,task,budget,permitted={'read','tree'},**kwargs))


def test_small_project_sources_have_real_evidence_and_no_verification_claim(tmp_path):
    (tmp_path/'module.py').write_text('VALUE=1\n')
    (tmp_path/'test_module.py').write_text('from module import VALUE\nassert VALUE==1\n')
    ws=workspace(tmp_path)
    result=prepare(ws)
    assert set(result['reads'])=={'module.py','test_module.py'}
    for record in result['records']:
        stored=json.loads((ws.directory/(record['evidence_id']+'.json')).read_text())
        path=record['args']['path']
        assert stored['origin']=='runtime_source_prefetch'
        assert stored['source_hashes'][path]==ws.file_digest(path)
        assert result['reads'][path]==record['output']
    assert ws.revision==0 and not ws.passed and not ws.changed


def test_large_inventory_only_prefetches_mentioned_files_with_utf8_byte_bound(tmp_path):
    for i in range(10):(tmp_path/f'unrelated{i}.py').write_text('UNRELATED\n')
    (tmp_path/'module.py').write_text('данные = '+repr('я'*4000))
    result=prepare(workspace(tmp_path),task='Inspect `module.py` for the reported issue.',budget=1600)
    assert set(result['reads'])=={'module.py'}
    assert len(json.dumps(result['reads'],ensure_ascii=False).encode())<=1600
    assert 'truncated' in result['reads']['module.py']
    assert 'UNRELATED' not in str(result)


def test_protected_and_outside_sources_are_never_loaded(tmp_path):
    root=tmp_path/'project';root.mkdir()
    (tmp_path/'outside.py').write_text('PRIVATE=1')
    (root/'link.py').symlink_to(tmp_path/'outside.py')
    (root/'.env').write_text('PRIVATE=2')
    (root/'module.py').write_text('PUBLIC=1')
    result=prepare(workspace(root),task='Read module.py, link.py and .env')
    assert set(result['reads'])=={'module.py'} and 'PRIVATE' not in str(result)


@pytest.mark.parametrize('options',[{'max_bytes':0,'permitted':{'read','tree'}},
    {'max_bytes':4800,'permitted':{'read'}},
    {'max_bytes':4800,'permitted':{'read','tree'},'cancelled':lambda:True}])
def test_prefetch_cannot_invent_authority_or_run_when_disabled(options):
    class Unavailable:
        async def execute(self,*args,**kwargs):raise AssertionError('must not access workspace')
    assert asyncio.run(prefetch_sources(Unavailable(),'task',**options))=={'reads':{},'hashes':{},'records':[]}


def test_remote_observations_use_backend_and_shared_recorder_not_local_source(tmp_path):
    (tmp_path/'module.py').write_text('HOST metadata must not appear')
    ws=workspace(tmp_path)
    view=copy.copy(ws)
    hashes={}
    async def execute(name,args,readonly=False):
        assert readonly
        if name=='tree':return {'ok':True,'output':'module.py'}
        assert name=='read'
        hashes['module.py']=hashlib.sha256(b'REMOTE=1').hexdigest()
        return {'ok':True,'output':'FILE module.py\nREMOTE=1'}
    view.execute=execute;view.file_digest=lambda name:hashes.get(name)
    assert ws.record({'ok':True,'output':'prior evidence'})=='1'
    result=prepare(view,record=ws.record)
    assert 'REMOTE=1' in result['reads']['module.py'] and 'HOST' not in str(result)
    assert result['records'][0]['evidence_id']=='2'
    assert ws.record({'ok':True,'output':'later evidence'})=='3'


def test_source_changed_during_read_is_not_prefetched(tmp_path):
    (tmp_path/'module.py').write_text('VALUE=1')
    ws=workspace(tmp_path);execute=ws.execute
    async def racing(name,args,**kwargs):
        result=await execute(name,args,**kwargs)
        if name=='read':(tmp_path/'module.py').write_text('VALUE=2')
        return result
    ws.execute=racing
    assert not prepare(ws)['reads'] and ws.next_id==0


def test_root_gets_initial_sources_without_buying_model_read_turns(tmp_path):
    (tmp_path/'module.py').write_text('VALUE=1\n')
    class Model:
        calls=0
        async def __call__(self,messages,schema,effort):
            self.calls+=1;packet=json.loads(messages[-1]['content'])
            assert 'VALUE=1' in packet['inspected_content']['module.py']
            assert packet['verified'] is False
            return schema.model_validate({'action':'finish','summary':'Inspected VALUE=1; no changes requested.'})
    model=Model()
    h=CodingHarness(tmp_path,model=model,config={'source_prefetch_bytes':4800,'checks':{},'learn':False,
        'planning':'adaptive','orchestration':{'enabled':True,'same_model':True,'critic':{'enabled':False}}})
    result=asyncio.run(h.run('Inspect module.py without modifying it.'))
    assert result['status']=='done' and model.calls==1 and not result['tools_used']
    assert not h.workspace.passed and h.workspace.revision==0


def test_prefetched_source_is_invalidated_before_using_a_stale_model_action(tmp_path):
    (tmp_path/'module.py').write_text('VALUE=1\n')
    class Model:
        calls=0
        async def __call__(self,messages,schema,effort):
            self.calls+=1;packet=json.loads(messages[-1]['content'])
            if self.calls==1:
                assert 'VALUE=1' in packet['inspected_content']['module.py']
                (tmp_path/'module.py').write_text('VALUE=3\n')
            else:
                assert 'module.py' not in packet.get('inspected_content',{})
                assert 'stale_sources' in str(packet['recent'])
            return schema.model_validate({'action':'finish','summary':'No edits; source changed externally.'})
    model=Model()
    result=asyncio.run(CodingHarness(tmp_path,model=model,config={
        'source_prefetch_bytes':4800,'max_turns':2,'checks':{},'learn':False}).run('Inspect module.py'))
    assert model.calls==2 and result['status']=='done'


def test_mathematical_task_does_not_prefetch_unrelated_repository(tmp_path):
    (tmp_path/'module.py').write_text('UNRELATED SOURCE')
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'domains':['mathematics'],'requirements':[
                    {'id':'r1','text':'Compute two plus two','validation':'model_evidence'}]})
            packet=json.loads(messages[-1]['content'])
            assert not packet.get('inspected_content') and 'UNRELATED SOURCE' not in str(messages)
            return schema.model_validate({'action':'plan','plan':['Compute 2+2 exactly.']})
    h=CodingHarness(tmp_path,model=Model(),config={'source_prefetch_bytes':4800,'max_turns':1,
        'checks':{},'learn':False,'orchestration':{'enabled':True,'auto_domains':True,
            'same_model':True,'critic':{'enabled':True,'active':False}}})
    asyncio.run(h.run('Compute 2+2.'))
    assert list(h.workspace.directory.glob('[1-9]*.json'))==[]


def test_readonly_worker_prefetch_has_distinct_shared_evidence_ids(tmp_path):
    from src.coding.control import SessionControl
    from src.coding.decision import Branch
    from src.coding.orchestration import Orchestrator
    from src.coding.supervisor import Supervisor
    (tmp_path/'module.py').write_text('VALUE=1\n')
    ws=workspace(tmp_path);assert ws.record({'ok':True,'output':'Parent evidence'})=='1'
    control=SessionControl()
    class Model:
        async def __call__(self,messages,schema,effort):
            memory=json.loads(messages[-1]['content'])['working_memory']
            assert 'VALUE=1' in memory['inspected_content']['module.py']
            assert '2' in memory['evidence_ids'] and memory['validation']['evidence_present']
            return schema.model_validate({'action':'finish','summary':'Inspected module.py.'})
    model=Model();h=CodingHarness(tmp_path,model=model,config={
        'source_prefetch_bytes':4800,'planning':'adaptive','learn':False});h.control=control
    o=Orchestrator(h,ws,Supervisor(ws.directory,control,model,False),
        {'same_model':True,'critic':{'enabled':False}})
    node=o.scheduler.create('main',Branch(mode='inspect',task='Inspect module.py',tools=['read','tree']))
    try:
        asyncio.run(o.run_mind(node))
        assert ws.record({'ok':True,'output':'Next parent evidence'})=='3'
        assert json.loads((ws.directory/'1.json').read_text())['output']=='Parent evidence'
    finally:o.branches.close()
