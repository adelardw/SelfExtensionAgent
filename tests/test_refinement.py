import pytest

from src.eval.refinement import Change, Proposal, ShadowRouter
from src.eval.refinement_cases import TASKS, cases, expected, score


def event(text='Use API v2', **kwargs):
    return {'id':'e1','text':text,'origin':'user','task_hint':'A',**kwargs}


def proposal(e, **kwargs):
    fields={'operation':'revise','task_id':'A','requirement_id':'r-api','text':'JSON API v2',
            'priority':'unchanged','depends_on':[],'delivery':'next_step',
            'source_quote':e['text'],'condition':''}
    fields.update(kwargs)
    return Proposal(changes=[Change(**fields)],question='')


def test_revision_preserves_other_constraints_and_tasks():
    r=ShadowRouter(TASKS); e=event()
    r.apply(e,proposal(e),r.snapshot())
    assert r.tasks['A']['requirements']['r-api']['text']=='JSON API v2'
    assert r.tasks['A']['requirements']['r-api']['revision']==2
    assert r.tasks['A']['requirements']['r-api']['priority']=='hard'
    assert r.tasks['A']['requirements']['r-python']==TASKS['A']['requirements']['r-python']
    assert r.tasks['A']['requirements']['r-latency']==TASKS['A']['requirements']['r-latency']
    assert r.tasks['B']==TASKS['B'] and TASKS['A']['revision']==4


def test_retraction_is_not_task_cancellation():
    r=ShadowRouter(TASKS);e=event('Drop latency cap')
    r.apply(e,proposal(e,operation='retract',requirement_id='r-latency'),r.snapshot())
    assert not r.tasks['A']['requirements']['r-latency']['active']
    assert r.tasks['A']['status']=='running'
    assert r.tasks['A']['requirements']['r-api']==TASKS['A']['requirements']['r-api']


def test_entire_plan_rejected_when_later_operation_is_invalid():
    r=ShadowRouter(TASKS);e=event();before=r.snapshot()
    p=proposal(e)
    p.changes.append(p.changes[0].model_copy(update={'task_id':'nonexistent'}))
    with pytest.raises(ValueError,match='unknown task'): r.apply(e,p,before)
    assert r.snapshot()==before and not r.events


def test_stale_proposal_cannot_override_newer_state():
    r=ShadowRouter(TASKS);e=event();before=r.snapshot()
    r.apply(e,proposal(e),before)
    updated=r.snapshot();late=event(id='e2')
    with pytest.raises(ValueError,match='stale'):r.apply(late,proposal(late),before)
    assert r.snapshot()==updated


def test_duplicate_delivery_is_idempotent_even_with_old_snapshot():
    r=ShadowRouter(TASKS);e=event();before=r.snapshot()
    first=r.apply(e,proposal(e),before);after=r.snapshot()
    second=r.apply(e,proposal(e),before)
    assert not first['duplicate'] and second['duplicate'] and r.snapshot()==after
    with pytest.raises(ValueError,match='different input'):
        r.apply(event('Actually API v3'),proposal(e),r.snapshot())
    assert r.snapshot()==after


@pytest.mark.parametrize('operation,dependencies,delivery,status',[
    ('new',[],'next_step','queued'),('dependent',['A'],'after_task','waiting')])
def test_independent_and_dependent_work_do_not_rewrite_existing_task(operation,dependencies,delivery,status):
    r=ShadowRouter(TASKS);e=event('Prepare a separate report')
    result=r.apply(e,proposal(e,operation=operation,task_id='',requirement_id='',
        depends_on=dependencies,delivery=delivery),r.snapshot())
    assert r.tasks['A']==TASKS['A'] and r.tasks['B']==TASKS['B']
    assert len(result['touched'])==1
    child=r.tasks[result['touched'][0]]
    assert child['status']==status and child['depends_on']==dependencies


def test_unresolved_condition_never_becomes_unconditional_action():
    r=ShadowRouter(TASKS);e=event('If p95 exceeds 200 ms, add a cache')
    result=r.apply(e,proposal(e,operation='add',requirement_id='',text='Add a cache',
        priority='hard',condition='If p95 exceeds 200 ms'),r.snapshot())
    assert r.tasks==TASKS and not result['touched']
    assert r.pending[0]['status']=='awaiting_condition'
    assert r.pending[0]['change']['condition']=='If p95 exceeds 200 ms'


def test_external_input_cannot_mutate_tasks_even_when_model_authorizes_it():
    r=ShadowRouter(TASKS);e=event(origin='external');before=r.snapshot()
    with pytest.raises(ValueError,match='external data'):r.apply(e,proposal(e),before)
    assert r.snapshot()==before
    r.apply(e,proposal(e,operation='context',task_id='',requirement_id='',delivery='none'),before)
    assert r.tasks==TASKS


def test_soft_preference_does_not_replace_hard_constraints():
    r=ShadowRouter(TASKS);e=event('Prefer the standard library')
    r.apply(e,proposal(e,operation='add',requirement_id='',priority='preferred',text=e['text']),r.snapshot())
    for rid,req in TASKS['A']['requirements'].items(): assert r.tasks['A']['requirements'][rid]==req
    new=[q for rid,q in r.tasks['A']['requirements'].items() if rid not in TASKS['A']['requirements']]
    assert len(new)==1 and new[0]['priority']=='preferred'


def test_clarification_never_silently_applies_half_a_compound_message():
    r=ShadowRouter(TASKS);e=event();before=r.snapshot()
    p=proposal(e);p.question='Which task?'
    with pytest.raises(ValueError,match='partial plan'):r.apply(e,p,before)
    assert r.snapshot()==before


@pytest.mark.parametrize('update,match',[
    ({'source_quote':'fabricated instruction'},'provenance'),
    ({'condition':'invented condition'},'condition'),
    ({'requirement_id':'unknown'},'unknown requirement'),
    ({'depends_on':['B']},'dependencies'),
    ({'delivery':'after_task'},'delivery')])
def test_model_cannot_invent_reference_or_delivery_authority(update,match):
    r=ShadowRouter(TASKS);e=event();before=r.snapshot()
    with pytest.raises(ValueError,match=match):r.apply(e,proposal(e,**update),before)
    assert r.snapshot()==before


def test_cancel_only_target_and_reject_later_implicit_resume():
    r=ShadowRouter(TASKS);e=event('Stop B immediately')
    r.apply(e,proposal(e,operation='cancel',task_id='B',requirement_id='',delivery='immediate'),r.snapshot())
    assert r.tasks['B']['status']=='cancelled' and r.tasks['A']==TASKS['A']
    later=event(id='e2')
    with pytest.raises(ValueError,match='not active'):
        r.apply(later,proposal(later,task_id='B',requirement_id='r-format'),r.snapshot())


def test_evaluator_rejects_missing_secondary_intent_and_wrong_priority():
    e=event();p=proposal(e)
    assert score(p,[expected('revise','A','r-api',contains=['v2'])])['exact']
    assert not score(p,[expected('revise','A','r-api'),expected('new')])['exact']
    assert not score(p,[expected('revise','A','r-api',priority='preferred')])['exact']
    assert len(cases())==18 and len({c['id'] for c in cases()})==18


def test_two_kinds_of_wait_are_kept_without_mutating_prerequisite():
    r=ShadowRouter(TASKS)
    e=event('After A, if checks fail, prepare a separate report')
    p=proposal(e,operation='dependent',task_id='',requirement_id='',depends_on=['A'],
               delivery='after_task',condition='if checks fail')
    result=r.apply(e,p,r.snapshot())
    child=r.tasks[result['touched'][0]]
    assert child['wait_for']=={'tasks':['A'],'predicate':'if checks fail'}
    assert child['status']=='waiting' and not r.pending
    assert r.tasks['A']==TASKS['A']


def test_conditional_independent_task_is_created_but_not_queued():
    r=ShadowRouter(TASKS);e=event('If it rains, create a reading plan')
    p=proposal(e,operation='new',task_id='',requirement_id='',condition='If it rains')
    result=r.apply(e,p,r.snapshot())
    child=r.tasks[result['touched'][0]]
    assert child['status']=='waiting'
    assert child['wait_for']=={'tasks':[],'predicate':'If it rains'}
    assert r.tasks['A']==TASKS['A'] and r.tasks['B']==TASKS['B']


def test_extra_context_and_extra_requirement_have_different_effect_scores():
    e=event('Stop B. Keep A running.')
    p=proposal(e,operation='cancel',task_id='B',requirement_id='',delivery='immediate')
    gold=[expected('cancel','B',delivery='immediate')]
    p.changes.append(proposal(e,operation='context',task_id='',requirement_id='',delivery='none').changes[0])
    graded=score(p,gold)
    assert graded['operation_match'] and not graded['exact'] and graded['unmatched_operations']==0
    p.changes[-1]=proposal(e,operation='add',requirement_id='',priority='hard',text='Keep running').changes[0]
    graded=score(p,gold)
    assert not graded['operation_match'] and graded['unmatched_operations']==1
    r=ShadowRouter(TASKS);r.apply(e,p,r.snapshot())
    # Structural acceptance is not a semantic correctness certificate.
    assert len(r.tasks['A']['requirements'])==len(TASKS['A']['requirements'])+1
