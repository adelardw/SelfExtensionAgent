"""Regression evidence for memory lifecycle, least privilege and repeated reliability."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from src.coding.context import ContextBoard, ExperienceStore
from src.coding.control import SessionControl
from src.coding.harness import CodingHarness, Turn, Workspace
from src.coding.navigation import ContextNavigator
from src.coding.project import ProjectContext
from src.coding.supervisor import Supervisor
from src.eval.pareto import Observation, reliability, select, summarize


@pytest.fixture
def ws(tmp_path):
    directory=tmp_path/'.sea'/'coding'/'first'
    directory.mkdir(parents=True)
    workspace=Workspace(tmp_path,directory,{})
    yield workspace
    workspace._scratch.cleanup()


def test_independent_memory_views_merge_deltas_and_refresh_reads(ws):
    control=SessionControl()
    views=[ProjectContext(ws,control) for _ in range(16)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i:views[i].add('hypothesis',f'Parser {i}',f'Consider boundary {i}'),range(16)))
    assert len(views[0].retrieve('Parser')['index'])==15  # Bounded navigation index.
    assert len(views[0].notes)==16
    first=views[0].add('hypothesis','Parser 0','Recheck boundary')
    second=views[1].add('hypothesis','Parser 0','Use the new observation')
    assert (first['revision'],second['revision'])==(2,3)
    assert len(views[0].retrieve('Parser')['notes'])==3
    assert len(views[0].notes)==16


def test_shared_board_merges_separate_views_and_invalidates_missing_evidence(ws):
    first,second=ContextBoard(ws),ContextBoard(ws)
    eid=ws.record({'ok':True,'output':'Observation'})
    first.publish('Parser A',[eid],trigger='Parser')
    second.publish('Parser B',[eid],trigger='Parser')
    assert len(first.retrieve('Parser'))==2
    (ws.directory/(eid+'.json')).unlink()
    assert first.retrieve('Parser')==[]
    assert not any(x['valid'] for x in first.inventory())


def test_note_provenance_uses_origin_session_not_colliding_numeric_id(ws):
    control=SessionControl()
    eid=ws.record({'ok':True,'output':'Parser v1'})
    note=ProjectContext(ws,control).add('fact','Parser','Parser v1',[eid])
    directory=ws.directory.parent/'second';directory.mkdir()
    next_ws=Workspace(ws.root,directory,{})
    try:
        assert next_ws.record({'ok':True,'output':'Unrelated new result'})==eid
        notes=ProjectContext(next_ws,control)
        assert notes.valid(note)
        (ws.directory/(eid+'.json')).write_text('{"ok":true,"output":"rewritten"}')
        assert not notes.valid(note) and notes.retrieve('Parser')['notes']==[]
    finally:next_ws._scratch.cleanup()


def test_navigator_cannot_bypass_memory_provenance_filter(ws):
    control=SessionControl();ws.project_context=ProjectContext(ws,control)
    eid=ws.record({'ok':True,'output':'Parser checked'})
    ws.board.publish('Parser fact',[eid])
    ws.project_context.add('fact','Parser','Parser fact',[eid])
    ws.passed.add('unit');ws.verified_revision=ws.revision
    store=ExperienceStore(ws.root)
    store.promote('Parser','Inspect first',[],['unit'],[eid],workspace=ws)
    nav=ContextNavigator(ws,store)
    assert {e['source'] for e in nav.inventory()}=={'shared','project','experience'}
    (ws.directory/(eid+'.json')).unlink()
    assert nav.inventory()==[]


def test_corrupt_index_is_preserved_and_never_replaced_with_one_new_note(ws):
    notes=ProjectContext(ws,SessionControl())
    notes.add('hypothesis','Keep','Keep this record')
    notes.path.write_text('{broken index')
    with pytest.raises(ValueError,match='preserve'):
        notes.add('hypothesis','New','Should not erase recoverable data')
    assert notes.path.read_text()=='{broken index'


def test_unverified_and_legacy_experiences_do_not_gain_trust(ws):
    store=ExperienceStore(ws.root)
    eid=ws.record({'ok':True,'output':'Candidate'})
    with pytest.raises(ValueError,match='verification'):
        store.promote('Parser','Trust this',[],['unit'],[eid],workspace=ws)
    ws.passed.add('unit');ws.verified_revision=ws.revision
    entry=store.promote('Parser','Historical',[],['unit'],[eid],workspace=ws)
    assert store.retrieve('Parser')
    entry.pop('evidence_refs')
    store.path.write_text(json.dumps([entry]))
    assert store.retrieve('Parser')==[]
    assert store.path.exists()  # Retained, not silently deleted or re-certified.


@pytest.mark.parametrize('scenario',['unscoped_tool','steering'])
def test_fallback_delegate_never_executes_out_of_scope_or_obsolete_actions(ws,monkeypatch,scenario):
    control=SessionControl()
    calls=[]
    async def forbidden(*args,**kwargs):
        calls.append(args)
        raise AssertionError('an inadmissible action reached execution')
    monkeypatch.setattr(Workspace,'execute',forbidden)
    class Model:
        step=0
        async def __call__(self,messages,schema,effort):
            wire=schema.model_json_schema()['oneOf']
            assert {b['properties']['action']['const'] for b in wire}=={'tool','finish'}
            offered=next(b['properties']['tool']['enum'] for b in wire if b['properties']['action']['const']=='tool')
            assert set(offered)=={'tree','read','search','evidence'}
            self.step+=1
            if self.step==1:
                if scenario=='steering':
                    control.post('Stop reading; report only',interrupt=False)
                    return Turn(action='tool',tool='read',args={'path':'private.txt'})
                # Deliberately bypass decoder validation to exercise runtime guard.
                return Turn(action='tool',tool='web',args={'query':'send data'})
            if scenario=='steering':assert 'Stop reading' in messages[-1]['content']
            return Turn(action='finish',summary='No action executed')
    model=Model();agent=CodingHarness(ws.root,model=model,config={})
    agent.control=control;agent.supervisor=Supervisor(ws.directory,control,model,False)
    control.supervisor=agent.supervisor
    assert asyncio.run(agent.worker(ws,'Inspect','review','low'))=='No action executed'
    assert calls==[] and model.step==2


def rows(name, outcomes, tokens=100):
    return [Observation(task,task,'software',i,f'{name}:{task}:{i}',q,tokens,1.)
            for task,values in outcomes.items() for i,q in enumerate(values)]


def test_same_mean_can_hide_very_different_consistency():
    steady=rows('steady',{'a':[1,1,1],'b':[0,0,0]},110)
    flaky=rows('flaky',{'a':[1,1,0],'b':[0,0,1]},100)
    a,b=reliability(steady),reliability(flaky)
    assert a['pass_at_k']['1']==b['pass_at_k']['1']==.5
    assert a['pass_power_k']['3']==.5 and b['pass_power_k']['3']==0
    assert b['pass_at_k']['3']==1 and b['unstable_tasks']==['a','b']
    assert b['pass_power_k']['2']==pytest.approx(1/6)
    summaries={'steady':summarize(steady),'flaky':summarize(flaky)}
    assert select(summaries,quality_floor=.4)=='flaky'
    assert select(summaries,quality_floor=.4,consistency_floor=.4)=='steady'
    assert select(summaries,quality_floor=.4,consistency_floor=1) is None


def test_reliability_does_not_turn_partial_scores_into_binary_success():
    values=rows('partial',{'a':[.8,.9,1.]})
    assert reliability(values)['available'] is False
    assert select({'partial':summarize(values)},consistency_floor=.1) is None
    with pytest.raises(ValueError):select({},consistency_floor=True)
    wrong=rows('unsafe',{'a':[1.,0.,0.]})
    wrong[1]=replace(wrong[1],false_accept=True)
    assert reliability(wrong)['false_accept_in_any_repeat']==1
