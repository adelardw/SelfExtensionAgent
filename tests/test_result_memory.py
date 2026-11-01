import asyncio
import json

import pytest

from src.coding.harness import Workspace,CodingHarness,Turn
from src.coding.result_memory import ResultMemory
from src.coding.request import compact_packet,fit_messages


def record(ws,memory,tool,args,output,version=0,ok=True):
    result={'ok':ok,'output':output}
    eid=ws.record(result)
    memory.observe(tool,args,result,eid,version)
    return eid


def test_exact_results_survive_rolling_window_and_resume_without_becoming_proof(tmp_path):
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    memory=ResultMemory(ws)
    first=record(ws,memory,'math',{'op':'simplify','expr':'110/74'},
                 json.dumps({'ok':True,'result':'55/37','resolved':True,'limitation':'Not a general proof.'}))
    for i in range(7):record(ws,memory,'math',{'op':'simplify','expr':str(i)},str(i))
    restored=ResultMemory(ws).packet(0)
    item=next(r for r in restored if r['evidence_id']==first)
    assert json.loads(item['output'])=={'result':'55/37','resolved':True,'limitation':'Not a general proof.'}
    assert item['args']['expr']=='110/74'
    assert not ws.passed and ws.verified_revision==-1
    assert not ResultMemory(ws).packet(1)


def test_results_are_deduplicated_bounded_and_only_reference_intact_real_evidence(tmp_path):
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    memory=ResultMemory(ws)
    args={'op':'simplify','expr':'1'}
    first=record(ws,memory,'math',args,'1')
    second=record(ws,memory,'math',dict(reversed(list(args.items()))),'1')
    args['expr']='2'
    rows=memory.packet(0)
    assert len(rows)==1 and rows[0]['evidence_id']==second and rows[0]['args']['expr']=='1'
    record(ws,memory,'math',{'expr':'2'},'bad',ok=False)
    record(ws,memory,'read',{'path':'x'},'irrelevant')
    assert len(memory.packet(0))==1
    (ws.directory/(second+'.json')).write_text('{"ok":true,"output":"changed"}')
    assert not memory.packet(0)
    with pytest.raises(ValueError,match='does not match'):
        memory.observe('math',{}, {'ok':True,'output':'not the record'},first,0)
    with pytest.raises(ValueError,match='recorded evidence'):
        memory.observe('math',{}, {'ok':True,'output':'x'},'../outside',0)
    for i in range(20):record(ws,memory,'experiment',{'code':'x'*2000+str(i)},'y'*3000)
    assert len(memory.entries)==12
    packet=memory.packet(0)
    assert sum(len(json.dumps(r,ensure_ascii=False)) for r in packet)<=5000
    assert all('excerpt' in r['args'] and 'excerpt' in r['output'] for r in packet)


def test_execution_observations_remain_historical_even_without_revision_change(tmp_path):
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    memory=ResultMemory(ws)
    record(ws,memory,'check',{'name':'unit'},'passed')
    item=memory.packet(0)[0]
    assert 'not current verification' in item['scope'] and not item['source_revision_changed']
    ws.revision+=1
    assert memory.packet(0)[0]['source_revision_changed']
    assert not ws.passed
    assert not ResultMemory(ws,'mind-1').packet(0)


def test_compaction_and_native_history_keep_actual_results_and_receipt_ids(tmp_path):
    from src.coding.native_tools import observation_messages,compile_tools
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    memory=ResultMemory(ws)
    args={'op':'simplify','expr':'2+2'}
    eid=record(ws,memory,'math',args,'4')
    packet={'task':'calculate','completed_results':memory.packet(0),
        'recent':[{'tool':'math','args':args,'evidence_id':eid,'ok':True,'output':'4'}]}
    compact=compact_packet(packet)
    assert compact['recent'][0]['output'].startswith('Result is included')
    messages=[{'role':'system','content':'policy'},{'role':'user','content':json.dumps(compact)}]
    _,receipt=fit_messages(messages,Turn,16384,4096)
    assert receipt['completed_results']==[eid]
    native,ids=observation_messages(messages,compile_tools(Turn))
    assert ids==[eid] and json.loads(native[2]['content'])['output']=='4'


def test_root_retains_computation_after_unrelated_reads(tmp_path,monkeypatch):
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    execute=Workspace.execute
    async def symbolic(ws,name,args,**kwargs):
        if name=='math':return {'ok':True,'output':'{"ok":true,"result":"4","resolved":true}'}
        return await execute(ws,name,args,**kwargs)
    monkeypatch.setattr(Workspace,'execute',symbolic)
    for i in range(5):(tmp_path/f'file{i}.txt').write_text(str(i))
    class Model:
        step=0
        async def __call__(self,messages,schema,effort):
            packet=json.loads(messages[-1]['content'])
            if self.step==6:
                assert not any(r.get('tool')=='math' for r in packet['recent'])
                assert json.loads(packet['completed_results'][0]['output'])['result']=='4'
            turns=[{'action':'tool','tool':'math','args':{'op':'simplify','expr':'2+2'}}]+[
                {'action':'tool','tool':'read','args':{'path':f'file{i}.txt'}} for i in range(5)]+[
                {'action':'finish','summary':'The observed computation returned 4; all inputs inspected.'}]
            value=schema.model_validate(turns[self.step]);self.step+=1;return value
    model=Model()
    result=asyncio.run(CodingHarness(tmp_path,model=model,config={'learn':False,'max_turns':10,'result_memory':True,
        'orchestration':{'enabled':True,'same_model':True,'critic':{'enabled':False}}}).run(
            'Compute 2+2 and inspect the five input files. Do not change files.'))
    assert result['status']=='done' and model.step==7
