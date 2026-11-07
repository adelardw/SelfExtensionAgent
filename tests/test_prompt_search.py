"""Offline selection/curation invariants; no network or local inference."""
import copy
import asyncio
import importlib
import json
from pathlib import Path

import pytest

from src.eval.prompt_search import Candidate, PromptDelta, Rule, curate, freeze, mutate, pareto, split_manifest
from src.eval.refinement_cases import cases
from src.eval.refinement import Change, Proposal, fingerprint
from src.eval.refinement_holdout import validation_cases, test_cases as heldout_cases


def record(cid='a', **updates):
    return {'case_id':cid,'split':'train','input':{'text':cid},'output':{'decision':'bad'},
            'feedback':'Unexpected new requirement','score':0,'complete':True,'infrastructure_error':False,
            'verifier_ok':True,'source_unchanged':True,**updates}


def splits():
    return {'train':cases(),'validation':validation_cases(),'test':heldout_cases()}


@pytest.mark.parametrize('update,reason',[
    ({'split':'test'},'not_training'),({'complete':False},'incomplete'),
    ({'infrastructure_error':True},'infrastructure_or_unknown'),
    ({'verifier_ok':False},'unreliable_verifier'),({'source_unchanged':False},'changed_or_unknown_source'),
    ({'score':float('nan')},'invalid_score'),({'score':True},'invalid_score'),
    ({'feedback':''},'missing_evidence'),({'output':{'api_key':'synthetic'}},'redacted')])
def test_noisy_or_sensitive_trajectory_is_excluded(update,reason):
    selected,excluded=curate([record(**update)],{'a'})
    assert selected==[] and excluded==[{'case_id':'a','reason':reason}]


def test_missing_metadata_fails_closed_and_success_is_retained():
    unknown=record('unknown');unknown.pop('infrastructure_error')
    rows=[record(str(i)) for i in range(5)]+[record('success',score=1),unknown]
    selected,excluded=curate(rows,{r['case_id'] for r in rows},limit=3)
    assert [r['case_id'] for r in selected]==['0','1','success']
    assert any(r['case_id']=='unknown' for r in excluded)


def test_conflicting_judgements_are_excluded_but_test_never_influences_curation():
    a=record();other=record(score=1)
    assert curate([a,other],{'a'})[0]==[]
    other['split']='test'
    assert len(curate([a,other],{'a'})[0])==1
    assert len(curate([record(score=1),record(score=1.0)],{'a'})[0])==2


def test_delta_retains_parent_and_requires_unchanged_training_provenance():
    records,_=curate([record()],{'a'})
    parent=Candidate('Immutable original permissions and instruction.')
    delta=PromptDelta(rules=[Rule(text='Keep execution state separate from acceptance criteria.',
                                 evidence=['a'],rationale='Extra requirement was added.')])
    child=mutate(parent,delta,records)
    assert child.prompt.startswith(parent.prompt) and child.parent==parent.id
    assert child.evidence_hash and child.id!=parent.id
    bad=copy.deepcopy(records);bad[0]['score']=1
    with pytest.raises(ValueError,match='uncurated'):mutate(parent,delta,bad)
    delta.rules[0].evidence=['unseen-test']
    with pytest.raises(ValueError,match='cite'):mutate(parent,delta,records)


def test_duplicate_rules_and_unbounded_accumulation_are_rejected():
    records,_=curate([record()],{'a'})
    rule=Rule(text='Do something.',evidence=['a'],rationale='Observed.')
    with pytest.raises(ValueError,match='duplicate'):mutate(Candidate('base'),PromptDelta(rules=[rule,rule]),records)
    with pytest.raises(ValueError,match='budget'):mutate(Candidate('x'*12000),PromptDelta(rules=[rule]),records)


def test_splits_reject_renamed_identical_input_despite_changed_attachment():
    data=splits();manifest=split_manifest(data)
    assert len(manifest['validation']['ids'])==6 and len(manifest['test']['ids'])==12
    duplicate=copy.deepcopy(data['train'][0]);duplicate['id']='renamed'
    duplicate['event']['id']='new-event';duplicate['event']['task_hint']='B'
    duplicate['event']['text']='  '+duplicate['event']['text'].upper()+'  '
    data['test'].append(duplicate)
    with pytest.raises(ValueError,match='duplicate'):split_manifest(data)


def test_frontier_keeps_complementary_solutions_and_rejects_incomplete_scores():
    assert pareto({'left':[1,0],'right':[0,1],'bad':[0,0]})==['left','right']
    assert pareto({'a':[1,1],'b':[1,1]})==['a','b']
    for invalid in ({'a':[]},{'a':[1],'b':[1,0]},{'a':[float('nan')]},{'a':[True]}):
        with pytest.raises(ValueError):pareto(invalid)


def test_freeze_uses_validation_only_and_does_not_promote_ties():
    manifest=split_manifest(splits());short=Candidate('short');long=Candidate('longer rules')
    candidates={c.id:c for c in (short,long)}
    vectors={k:[1]*6 for k in candidates}
    frozen=freeze(candidates,vectors,manifest['validation']['ids'],manifest)
    assert frozen['selected']==short.id and frozen['production_promoted'] is False
    vectors[short.id][0]=0
    assert frozen['vectors'][short.id][0]==1
    with pytest.raises(ValueError,match='identities'):
        freeze(candidates,vectors,manifest['test']['ids'][:6],manifest)


def gold_proposal(case):
    changes=[]
    for g in case['expected']:
        changes.append(Change(**{k:v for k,v in g.items() if k not in {'conditional','contains'}},
            text=' '.join(g['contains']) or 'Assignment',source_quote=case['event']['text'],
            condition=case['event']['text'] if g['conditional'] else ''))
    return Proposal(changes=changes,question='Which task?' if changes[0].operation=='clarify' else '')


def prepare_archive(path):
    path.mkdir();rows=cases()
    (path/'cases.json').write_text(json.dumps(rows))
    (path/'protocol.json').write_text(json.dumps({'cases_sha256':fingerprint(rows)}))
    (path/'result.json').write_text(json.dumps({'results':[
        {'id':c['id'],'proposal':gold_proposal(c).model_dump()} for c in rows]}))


def test_campaign_freezes_before_test_excludes_gold_and_never_calls_runtime(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    script=importlib.import_module('refinement_search')
    archive=tmp_path/'archive';prepare_archive(archive)
    output=tmp_path/'trial'
    known={c['id']:c for items in splits().values() for c in items}
    seen=[]

    class FakeModel:
        def __init__(self,directory,shared,max_calls):
            self.shared=shared;self.config={};assert max_calls==6
        def usage(self):return {'calls':self.shared['calls'],'tokens':60,'cost':0}
        async def __call__(self,messages,schema,effort):
            self.shared['calls']+=1
            data=json.loads(messages[1]['content']);seen.append(data)
            if schema is PromptDelta:
                assert all(r['case_id'] in {c['id'] for c in cases()} for r in data['training_records'])
                return PromptDelta(rules=[Rule(text='Preserve intent.',evidence=['add'],rationale='Observed.')])
            batch=data['independent_cases']
            assert all(set(c)=={'case_id','tasks','event'} for c in batch)
            if batch[0]['case_id'].startswith('test-'):
                selected=json.loads((output/'selection.json').read_text())
                assert selected['prompt']==messages[0]['content'] and not selected['production_promoted']
            else: assert not (output/'selection.json').exists()
            return schema(decisions=[{'case_id':c['case_id'],'proposal':gold_proposal(known[c['case_id']])} for c in batch])

    monkeypatch.setattr(script,'CodexInference',FakeModel)
    asyncio.run(script.search(output,archive))
    result=json.loads((output/'result.json').read_text())
    assert result['calls']==6 and len(seen)==6 and result['test_verified']==12
    assert result['selected']=='v1' and result['cost_usd'] is None
    assert result['source_unchanged'] and not result['production_promoted']
    with pytest.raises(FileExistsError):asyncio.run(script.search(output,archive))


def test_infrastructure_failure_aborts_selection_and_preserves_receipt(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    script=importlib.import_module('refinement_search')
    archive=tmp_path/'archive';prepare_archive(archive)
    class Unavailable:
        def __init__(self,directory,shared,max_calls):self.config={}
        async def __call__(self,*args):raise TimeoutError('provider unavailable')
    monkeypatch.setattr(script,'CodexInference',Unavailable)
    output=tmp_path/'trial'
    with pytest.raises(TimeoutError):asyncio.run(script.search(output,archive))
    assert not (output/'selection.json').exists()
    assert not (output/'result.json').exists()
    assert json.loads((output/'incomplete.json').read_text())['status']=='incomplete'
