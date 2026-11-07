"""Completion must see attributable source, not just the first deployment files."""
import asyncio
import json

from src.coding.control import SessionControl
from src.coding.critic import ContextCritic
from src.coding.harness import Workspace, CodingHarness
from src.coding.minds import MindScheduler
from src.coding.supervisor import Supervisor


def test_review_covers_files_fairly_and_cites_only_transmitted_evidence(tmp_path):
    (tmp_path/'README.md').write_text('intro\n'*2500+'RESTORE_WITH_INTEGRITY\n')
    for i in range(12): (tmp_path/f'module{i}.py').write_text(f'VALUE_{i} = {i}\n')
    directory=tmp_path/'.sea';directory.mkdir()
    control=SessionControl();ws=Workspace(tmp_path,directory,{})
    ws.changed.update(['README.md',*[f'module{i}.py' for i in range(12)]])
    unseen=ws.record({'ok':True,'output':'old observation not shown to reviewer'})
    class Model:
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Review all modules','validation':'model_evidence'}]})
            packet=json.loads(messages[-1]['content']);sources=packet['current_source']
            assert set(sources)==ws.changed
            assert 'RESTORE_WITH_INTEGRITY' in sources['README.md']['text']
            assert sources['README.md']['truncated']
            assert sum(len(s['text']) for s in sources.values())<=8000
            assert all(s['sha256']==ws.file_digest(name) for name,s in sources.items())
            visible={s['evidence_id'] for s in sources.values()}|{e['id'] for e in packet['evidence']}|{packet['candidate_answer']['id']}
            assert set(packet['valid_evidence_ids'])==visible
            return schema.model_validate({'aligned':True,'ready':False,'criteria':{
                'r1':{'status':'uncertain','evidence':[sources['README.md']['evidence_id']]}},
                'issues':['The README middle was not reviewed.'],'next_mode':'inspect'})
    model=Model();supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task='Review all modules';node.memory.summary='Candidate'
    node.memory.evidence=[unseen]+[ws.record({'ok':True,'output':f'observation {i}'}) for i in range(20)]
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'max_calls':2,'source_context_chars':8000})
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert not result.ready


def test_resume_retains_current_reads_but_not_stale_sources_or_checks(tmp_path):
    for name in ('keep.py','changed.py'): (tmp_path/name).write_text('ORIGINAL = 1\n')
    class ReadModel:
        step=0
        async def __call__(self,messages,schema,effort):
            name=('keep.py','changed.py')[self.step];self.step+=1
            return schema.model_validate({'action':'tool','tool':'read','args':{'path':name}})
    config={'planning':'adaptive','checks':{},'max_turns':2,'learn':False}
    first=CodingHarness(tmp_path,model=ReadModel(),config=config)
    asyncio.run(first.run('Read both files.',session_id='resume-source'))
    (tmp_path/'changed.py').write_text('UPDATED = 2\n')
    class ResumeModel:
        async def __call__(self,messages,schema,effort):
            packet=json.loads(messages[-1]['content'])
            assert 'ORIGINAL = 1' in packet['inspected_content']['keep.py']
            assert 'changed.py' not in packet['inspected_content']
            assert not packet['verified']
            return schema.model_validate({'action':'finish','summary':'Unchanged read retained; changed file requires a new read.'})
    second=CodingHarness(tmp_path,model=ResumeModel(),config=config)
    result=asyncio.run(second.run('Read both files.',session_id='resume-source',resume=True))
    assert result['status']=='done'


def test_watchdog_recognizes_read_and_check_progress_without_file_edits():
    from src.coding.progress import ProgressWatchdog
    watch=ProgressWatchdog(3)
    for epoch in range(10):
        assert not watch.observe({'revision':0,'evidence_epoch':epoch,'passed':[]})
    final={'revision':0,'evidence_epoch':9,'passed':['unit']}
    assert not watch.observe(final)
    assert not watch.observe(final)
    assert not watch.observe(final)
    assert watch.observe(final)


def test_source_coverage_cannot_read_secrets_or_escape_workspace(tmp_path):
    from src.coding.review_context import current_sources
    root=tmp_path/'project';root.mkdir();directory=root/'.sea';directory.mkdir()
    (tmp_path/'private.txt').write_text('OUTSIDE_SECRET')
    (root/'.env').write_text('PRIVATE_KEY=DO_NOT_TRANSMIT')
    (root/'safe.py').write_text('VALUE=1\n')
    ws=Workspace(root,directory,{})
    sources,coverage=asyncio.run(current_sources(ws,['.env','../private.txt','safe.py']))
    assert set(sources)=={'safe.py'}
    serialized=json.dumps([sources,coverage])
    assert 'OUTSIDE_SECRET' not in serialized and 'DO_NOT_TRANSMIT' not in serialized
    assert sum(r['status']=='unavailable' for r in coverage['files'])==2


def test_source_capture_race_never_becomes_current_evidence(tmp_path):
    from src.coding.review_context import current_sources
    directory=tmp_path/'.sea';directory.mkdir()
    (tmp_path/'a.py').write_text('BEFORE=1\n')
    class RacingWorkspace(Workspace):
        async def source_text(self,name,limit=4000):
            text=await super().source_text(name,limit)
            self.path(name).write_text('AFTER=2\n')
            return text
    ws=RacingWorkspace(tmp_path,directory,{})
    sources,coverage=asyncio.run(current_sources(ws,['a.py']))
    assert not sources
    assert coverage['files']==[{'path':'a.py','status':'changed_during_capture'}]


def test_source_snapshot_ids_do_not_evict_observations_or_invalidate_review_cache(tmp_path):
    directory=tmp_path/'.sea';directory.mkdir()
    ws=Workspace(tmp_path,directory,{})
    for index in range(14):
        name=f'module{index}.py';(tmp_path/name).write_text(f'VALUE={index}\n');ws.changed.add(name)
    eid=ws.record({'ok':True,'output':'A recorded behavioral observation'})
    class Model:
        reviews=0
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[{'id':'r1','text':'Review behavior','validation':'model_evidence'}]})
            self.reviews+=1
            data=json.loads(messages[-1]['content'])
            assert eid in data['valid_evidence_ids']
            return schema.model_validate({'aligned':True,'ready':True,'criteria':{
                'r1':{'status':'supported','evidence':[eid]}},'issues':[],'next_mode':'verify'})
    control=SessionControl();model=Model();supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},ws).nodes['main']
    node.task='Review behavior';node.memory.summary='Observed behavior';node.memory.evidence=[eid]
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'max_calls':4})
    assert asyncio.run(critic.assess(node,ws,'finish')).ready
    assert asyncio.run(critic.assess(node,ws,'finish')).ready
    assert model.reviews==1
