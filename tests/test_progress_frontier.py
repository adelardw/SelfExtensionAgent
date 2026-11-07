import asyncio
import json
import pytest

from src.coding.harness import Turn,Workspace,CodingHarness
from src.coding.progress import ActionFrontier
from src.coding.orchestration import scoped_schema
from src.coding.native_tools import compile_tools,decode_call


def test_failure_focus_survives_unrelated_reads_resume_and_marks_changed_source(tmp_path):
    from src.coding.progress import FailureFocus
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    (tmp_path/'module.py').write_text('VALUE=1\n')
    focus=FailureFocus(ws)
    failed={'ok':False,'output':'expected 2, observed 1'}
    eid=ws.record(failed)
    focus.observe(ws,'check',{'name':'unit'},failed,eid,0,['module.py'])
    focus.observe(ws,'read',{'path':'other.py'},{'ok':True,'output':'other'},'2',0)
    restored=FailureFocus(ws)
    assert restored.packet(ws,0)['evidence_id']==eid
    assert restored.packet(ws,0)['status']=='observed_failure'
    (tmp_path/'module.py').write_text('VALUE=2\n')
    assert restored.packet(ws,0)['source_changed_since_failure']
    assert not restored.packet(ws,1)  # Old operator/task scope is not current.
    restored.observe(ws,'check',{'name':'unit'},{'ok':True,'output':'passed'},'3',0)
    assert not FailureFocus(ws).packet(ws,0)


def test_current_failure_does_not_leave_context_when_recent_observations_roll(tmp_path,monkeypatch):
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    for i in range(5):(tmp_path/f'file{i}.txt').write_text(str(i))
    execute=Workspace.execute
    async def with_failure(ws,name,args,**kwargs):
        if name=='experiment':return {'ok':False,'output':'ZeroDivisionError: division by zero'}
        return await execute(ws,name,args,**kwargs)
    monkeypatch.setattr(Workspace,'execute',with_failure)
    class Model:
        step=0
        failure_id=None
        async def __call__(self,messages,schema,effort):
            data=json.loads(messages[-1]['content'])
            if self.step==2:
                actions={b['properties']['action']['const'] for b in schema.model_json_schema()['oneOf']}
                assert 'plan' not in actions  # No new evidence since the recorded plan.
                self.failure_id=data['active_failure']['evidence_id']
            if self.step>=2:
                assert data['active_failure']['evidence_id']==self.failure_id
                assert 'ZeroDivisionError' in data['active_failure']['output']
                assert not data['active_failure']['source_changed_since_failure']
            if self.step==7:
                assert set(data['inspected_content'])=={f'file{i}.txt' for i in range(5)}
            turns=[{'action':'plan','plan':['Observe the expected error; inspect the five inputs']},
                   {'action':'tool','tool':'experiment','args':{'code':'1/0'}}]
            turns += [{'action':'tool','tool':'read','args':{'path':f'file{i}.txt'}} for i in range(5)]
            turns += [{'action':'finish','summary':'Observed the expected division-by-zero error and inspected inputs.'}]
            value=schema.model_validate(turns[self.step]);self.step+=1;return value
    model=Model()
    result=asyncio.run(CodingHarness(tmp_path,model=model,config={'learn':False,'max_turns':10,
        'orchestration':{'enabled':True,'same_model':True,'critic':{'enabled':False}}}).run(
            'Observe the division-by-zero error, then inspect the five input files. No edits needed.'))
    assert result['status']=='done' and model.step==8


def test_source_window_is_byte_bounded_and_marks_clipped_observations():
    from src.coding.progress import source_window
    reads={f'f{i}.py':'small source' for i in range(5)}
    assert source_window(reads)==reads
    reads['latest.py']='строка\n"quoted"\n'*2000
    packet=source_window(reads)
    assert len(json.dumps(packet,ensure_ascii=False).encode())<=4800
    assert 'excerpt truncated' in packet['latest.py']
    assert reads['latest.py'].startswith(packet['latest.py'].split('\n[excerpt truncated')[0])


def test_multiple_read_ranges_remain_visible_without_inventing_gaps():
    from src.coding.progress import merge_read_fragments
    schema='FILE app.py · lines 10–11 of 200\nTABLE videos\nPRIMARY KEY id'
    helper='FILE app.py · lines 80–81 of 200\ndef require_admin(request):\n    return session'
    combined=merge_read_fragments(schema,helper)
    assert combined.startswith(helper) and schema in combined
    assert 'previous inspected range' in combined
    # Reading a superset replaces the covered fragment, not unrelated evidence.
    larger='FILE app.py · lines 79–82 of 200\n# auth\ndef require_admin(request):\n    return session\n# end'
    merged=merge_read_fragments(combined,larger)
    assert merged.count('def require_admin')==1 and 'TABLE videos' in merged
    assert merge_read_fragments(merged,larger)==merged


def test_read_cache_budget_and_different_source_version_fail_closed():
    from src.coding.progress import merge_read_fragments
    old='FILE a.py · lines 1–2 of 200\nold schema\nold helper'
    new='FILE a.py · lines 90–91 of 201\nnew schema\nnew helper'
    assert 'old schema' not in merge_read_fragments(old,new)
    other=new.replace('a.py','b.py')
    assert 'old schema' not in merge_read_fragments(old,other)
    large='FILE a.py · lines 3–90 of 200\n'+'source\n'*100
    cache=merge_read_fragments(old,large,limit=300)
    assert len(cache)<=300 and 'excerpt truncated' in cache


def test_actor_can_use_two_parts_of_one_file_in_the_next_decision(tmp_path):
    lines=['# filler']*120;lines[9]='TABLE_NAME = "videos"';lines[99]='def require_admin(): return True'
    (tmp_path/'module.py').write_text('\n'.join(lines)+'\n')
    class Model:
        step=0
        async def __call__(self,messages,schema,effort):
            data=json.loads(messages[-1]['content'])
            if self.step==3:
                observed=data['inspected_content']['module.py']
                assert 'TABLE_NAME = "videos"' in observed and 'def require_admin' in observed
            turns=[{'action':'plan','plan':['Inspect both relevant parts']},
                   {'action':'tool','tool':'read','args':{'path':'module.py','start':10,'end':10}},
                   {'action':'tool','tool':'read','args':{'path':'module.py','start':100,'end':100}},
                   {'action':'finish','summary':'Both declarations were inspected.'}]
            turn=turns[self.step];self.step+=1
            return schema.model_validate(turn)
    result=asyncio.run(CodingHarness(tmp_path,model=Model(),config={'learn':False,'max_turns':5,'checks':{}}).run(
        'Inspect the table name and authorization helper in module.py. Do not edit files.'))
    assert result['status']=='done'


def test_source_window_prioritizes_recent_reads_and_discards_stale_worker_text(tmp_path):
    from src.coding.minds import WorkingMemory
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    mem=WorkingMemory()
    for name in ('a.py','b.py','c.py'):
        (tmp_path/name).write_text(name)
        mem.reads[name]=name;mem.files[name]=ws.file_digest(name)
    assert set(mem.packet(ws)['inspected_content'])=={'a.py','b.py','c.py'}
    (tmp_path/'b.py').write_text('changed')
    assert 'b.py' not in mem.packet(ws)['inspected_content']


def test_actor_receives_current_source_immediately_after_edit(tmp_path):
    import sys
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    (tmp_path/'value.py').write_text('VALUE=1\n')
    class Model:
        step=0
        async def __call__(self,messages,schema,effort):
            data=json.loads(messages[-1]['content'])
            if self.step==3:
                assert 'VALUE=2' in data['inspected_content']['value.py']
                assert 'VALUE=1' not in data['inspected_content']['value.py']
            turns=[{'action':'plan','plan':['Read, update and verify VALUE']},
                {'action':'tool','tool':'read','args':{'path':'value.py'}},
                {'action':'tool','tool':'edit','args':{'path':'value.py','old':'VALUE=1','new':'VALUE=2'}},
                {'action':'tool','tool':'check','args':{'name':'unit'}},
                {'action':'finish','summary':'VALUE updated and checked'}]
            value=schema.model_validate(turns[self.step]);self.step+=1;return value
    result=asyncio.run(CodingHarness(tmp_path,model=Model(),config={'learn':False,
        'checks':{'unit':[sys.executable,'-B','-c','from value import VALUE; assert VALUE==2']},
        'orchestration':{'enabled':True,'same_model':True,'critic':{'enabled':False}}}).run('Set VALUE to 2 and verify'))
    assert result['status']=='done'


@pytest.mark.parametrize('protocol',['json','native'])
@pytest.mark.parametrize('tool,args,corrected',[
    ('math',{'op':'hypergeom','N':9,'K':5,'n':3},{'op':'sum','expr':'k','var':'k','bounds':[0,2]}),
    ('read',{'path':'missing.py'},{'path':'present.py'}),
    ('read',{'path':'present.py','start':1,'end':3},{'path':'present.py','start':4,'end':6}),
])
def test_repeat_blocks_exact_call_but_allows_corrected_arguments(protocol,tool,args,corrected):
    frontier=ActionFrontier();frontier.sync((1,0,0))
    one=Turn(action='tool',tool=tool,args=args)
    two=Turn(action='tool',tool=tool,args=dict(reversed(list(args.items()))))
    frontier.note(one);assert frontier.allowed(two)
    frontier.note(two);assert not frontier.allowed(one)
    schema=scoped_schema(Turn,{'math','read'},False,False,frontier.actions,blocked_calls=frontier.calls.values())
    definitions=compile_tools(schema)
    assert {d['function']['name'] for d in definitions}=={'math','read','finish'}
    def parse(arguments):
        value=({'action':'tool','tool':tool,'args':arguments} if protocol=='json' else
               decode_call({0:{'name':tool,'arguments':json.dumps(arguments)}},definitions))
        return schema.model_validate(value)
    with pytest.raises(ValueError,match='Unchanged repeated call'):parse(two.args)
    changed=parse(corrected)
    assert frontier.allowed(changed) and changed.args==corrected
    assert frontier.packet()['blocked_calls']==[{'tool':tool,'args':args}]
    with pytest.raises(ValueError,match='Tool is not available'):
        schema.model_validate({'action':'tool','tool':'write','args':{'path':'x','content':'y'}})
    frontier.sync((1,0,0));assert not frontier.allowed(one)
    frontier.sync((2,0,0));assert frontier.allowed(one)
    assert not frontier.packet()


@pytest.mark.parametrize('epoch',[(2,0,0),(1,1,0),(1,0,1)])
def test_observation_source_or_operator_change_resets_exact_call_restrictions(epoch):
    frontier=ActionFrontier();frontier.sync((1,0,0))
    turn=Turn(action='tool',tool='read',args={'path':'module.py'})
    frontier.note(turn);frontier.note(turn)
    schema=scoped_schema(Turn,{'read'},blocked_calls=frontier.calls.values())
    frontier.sync(epoch)
    assert frontier.allowed(turn) and not frontier.packet()
    # A schema already attached to an in-flight request retains its own scope.
    with pytest.raises(ValueError,match='Unchanged repeated call'):
        schema.model_validate(turn.model_dump())


def test_root_can_read_correct_path_after_two_identical_failed_reads(tmp_path):
    (tmp_path/'present.py').write_text('VALUE=2\n')
    class Model:
        step=0
        async def __call__(self,messages,schema,effort):
            packet=json.loads(messages[-1]['content'])
            if self.step==2:
                assert packet['recovery']['blocked_calls']==[{'tool':'read','args':{'path':'missing.py'}}]
            turns=[{'action':'tool','tool':'read','args':{'path':'missing.py'}}]*2+[
                {'action':'tool','tool':'read','args':{'path':'present.py'}},
                {'action':'finish','summary':'Read the corrected path; VALUE is 2.'}]
            value=schema.model_validate(turns[self.step]);self.step+=1;return value
    model=Model()
    result=asyncio.run(CodingHarness(tmp_path,model=model,config={'learn':False,'max_turns':6,
        'orchestration':{'enabled':True,'same_model':True,'critic':{'enabled':False}}}).run(
            'Inspect the value without changing files.'))
    assert result['status']=='done' and model.step==4
    assert result['tools_used']==['read','read','read']


def test_identical_rejected_answer_cannot_consume_every_turn():
    frontier=ActionFrontier();frontier.sync(0)
    candidate=Turn(action='finish',summary='same unsupported answer')
    frontier.note(candidate);frontier.note(candidate)
    schema=scoped_schema(Turn,{'read'},False,False,frontier.actions)
    assert {b['properties']['action']['const'] for b in schema.model_json_schema()['oneOf']}=={'tool'}


def test_workspace_overview_uses_backend_not_local_metadata(tmp_path):
    class Remote(Workspace):
        calls=0
        async def execute(self,name,args,readonly=False):
            assert name=='tree' and readonly
            self.calls+=1
            return {'ok':True,'output':'remote/main.py\nremote/test.py'}
    ws=Remote(tmp_path,tmp_path/'.sea',{})
    (tmp_path/'metadata-only.py').write_text('not the real project')
    result=asyncio.run(ws.overview())
    assert result['paths']==['remote/main.py','remote/test.py']
    assert asyncio.run(ws.overview())==result and ws.calls==1
    ws.revision+=1;asyncio.run(ws.overview());assert ws.calls==2


def test_invalid_starting_contract_keeps_a_resumable_checkpoint(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):return schema.model_validate({'requirements':[]})
    h=CodingHarness(tmp_path,model=Model(),config={'orchestration':{'enabled':True,'auto_domains':True,
        'same_model':True,'critic':{'enabled':True,'max_calls':1}}})
    result=asyncio.run(h.run('Solve a problem'))
    assert result['status']=='paused' and 'ValidationError' in result['final_answer']
    assert json.loads(open(result['checkpoint']).read())['query']=='Solve a problem'


def test_finish_is_offered_only_after_explicit_mechanical_obligations(tmp_path):
    import sys
    from src.runtime import hitl
    class Model:
        step=0
        async def __call__(self,messages,schema,effort):
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[
                    {'id':'r1','text':'Add tests','validation':'new_or_updated_tests','source_quote':'Add tests'},
                    {'id':'r2','text':'run checks','validation':'registered_checks','source_quote':'run checks'}]})
            packet=json.loads(messages[-1]['content'])
            if 'oneOf' not in schema.model_json_schema():
                assert self.step==4
                return schema.model_validate({'aligned':True,'ready':True,'criteria':[
                    {'requirement':'r1','status':'supported','evidence':packet['valid_evidence_ids'][-1:]},
                    {'requirement':'r2','status':'supported','evidence':packet['valid_evidence_ids'][-1:]}],
                    'issues':[],'next_mode':'verify'})
            actions={b['properties']['action']['const'] for b in schema.model_json_schema()['oneOf']}
            assert ('finish' in actions)==(self.step==3)
            assert bool(packet.get('completion_blockers'))==(self.step<3)
            if self.step<2:
                assert 'existing test file' in packet['completion_guidance']
                assert 'Running existing tests does not add tests' in packet['completion_guidance']
                assert 'existing test file' not in packet['next_action_hint']
                assert 'not a work order' in packet['completion_guidance']
            elif self.step==2:
                assert 'tool=check' in packet['completion_guidance'] and '"unit"' in packet['completion_guidance']
            else:
                assert 'completion_guidance' not in packet
            turns=[{'action':'plan','plan':['Add test and run checks']},
                {'action':'tool','tool':'write','args':{'path':'test_new.py','content':'assert 1+1==2\n'}},
                {'action':'tool','tool':'check','args':{'name':'unit'}},
                {'action':'finish','summary':'Tests added and checks passed'}]
            result=schema.model_validate(turns[self.step]);self.step+=1;return result
    hitl.set_work_mode('auto')
    h=CodingHarness(tmp_path,model=Model(),config={'learn':False,
        'checks':{'unit':[sys.executable,'-B','test_new.py']},
        'orchestration':{'enabled':True,'auto_domains':True,'same_model':True,
                         'critic':{'enabled':True,'active':False}}})
    result=asyncio.run(h.run('Add tests and run checks'))
    assert result['status']=='done'


def test_failed_new_test_requires_diagnosis_before_rerunning_check(tmp_path):
    import sys
    from src.runtime import hitl
    (tmp_path/'numbers_impl.py').write_text('def double(x): return 2*x\n')
    class Model:
        step=0
        async def __call__(self,messages,schema,effort):
            packet=json.loads(messages[-1]['content'])
            if schema.__name__=='TaskContract':
                return schema.model_validate({'requirements':[
                    {'id':'r1','text':'Add regression tests','validation':'new_or_updated_tests','source_quote':'Add regression tests'},
                    {'id':'r2','text':'run checks','validation':'registered_checks','source_quote':'run checks'}]})
            if 'oneOf' not in schema.model_json_schema():
                return schema.model_validate({'aligned':True,'ready':True,'criteria':[
                    {'requirement':rid,'status':'supported','evidence':packet['valid_evidence_ids'][-1:]}
                    for rid in ('r1','r2')],'issues':[],'next_mode':'verify'})
            if self.step==3:
                assert packet['failed_checks']==['unit']
                assert 'Diagnose' in packet['next_action_hint']
                assert 'newly added test' in packet['completion_guidance']
                assert not any(b['properties']['action']['const']=='finish' for b in schema.model_json_schema()['oneOf'])
            turns=[{'action':'plan','plan':['Test the specified double behavior and run checks']},
                {'action':'tool','tool':'write','args':{'path':'test_double.py',
                    'content':'from numbers_impl import double\nassert double(3)==5\n'}},
                {'action':'tool','tool':'check','args':{'name':'unit'}},
                {'action':'tool','tool':'edit','args':{'path':'test_double.py','old':'double(3)==5','new':'double(3)==6'},
                    'summary':'The new expectation was wrong: the task specifies twice x, so double(3) is 6.'},
                {'action':'tool','tool':'check','args':{'name':'unit'}},
                {'action':'finish','summary':'Added a regression test for twice x; the registered check passed.'}]
            result=schema.model_validate(turns[self.step]);self.step+=1;return result
    hitl.set_work_mode('auto')
    h=CodingHarness(tmp_path,model=Model(),config={'learn':False,
        'checks':{'unit':[sys.executable,'-B','test_double.py']},
        'orchestration':{'enabled':True,'auto_domains':True,'same_model':True,
                         'critic':{'enabled':True,'active':False}}})
    result=asyncio.run(h.run('Add regression tests for double(x), which returns twice x, and run checks'))
    assert result['status']=='done' and result['checks_passed']==['unit']
    assert (tmp_path/'numbers_impl.py').read_text()=='def double(x): return 2*x\n'
