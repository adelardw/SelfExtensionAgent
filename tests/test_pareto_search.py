"""Selection, provenance, budget and staged-search behavior; no model calls."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.eval.pareto import Observation, frontier, grid, promote, select, summarize
from src.eval.harness_search import (
    SearchPlan, campaign, compatible, inventory, load_stage, observations, save_json,
)


def outcomes(name, quality=1., tokens=100., tasks=('a', 'b', 'c')):
    return [Observation(task, task, ('software', 'mathematics', 'algorithms')[i],
                        r, f'{name}:{task}:{r}', quality, tokens, 10.)
            for i, task in enumerate(tasks) for r in range(3)]


def test_promotion_preserves_temporary_cost_increase_but_deployment_has_own_floor():
    old = outcomes('old', .5, 100.)
    new = outcomes('new', .9, 140.)
    decision = promote(old, new, draws=1000)
    assert decision['promoted'] and decision['support'] == 1
    assert select({'new': summarize(new)}, max_tokens=120) is None
    assert not promote(old, outcomes('excessive', .9, 170.), draws=1000)['promoted']


def test_cheaper_failure_does_not_beat_quality_and_equal_results_do_not_improve():
    parent = outcomes('old', 1., 100.)
    assert not promote(parent, outcomes('failed', 0., 1.), draws=1000)['promoted']
    assert not promote(parent, outcomes('same', 1., 100.), draws=1000)['promoted']
    assert promote(parent, outcomes('cheaper', 1., 70.), draws=1000)['promoted']
    assert not promote(outcomes('free', .5, 0.), outcomes('paid', 1., 1.), draws=1000)['promoted']


def test_bootstrap_preserves_quality_cost_pairing_and_is_reproducible():
    parent = outcomes('old', .8, 100.)
    child = [replace(r, quality=[0., 1., 1.][r.repeat], tokens=[500., 30., 30.][r.repeat])
             for r in outcomes('new')]
    first = promote(parent, child, draws=2000, seed=7)
    assert first == promote(parent, child, draws=2000, seed=7)
    assert 0 < first['support'] < 1
    assert first['uncertainty_scope'].startswith('execution variation')


def test_loose_bootstrap_or_gate_cannot_promote_identical_empirical_distributions():
    parent = [Observation('a', 'a', 'software', i, f'parent:{i}', q, c, 1.)
              for i, (q, c) in enumerate([(.8, 80.), (.9, 90.), (1., 100.)])]
    child = [replace(x, run_id=f'child:{i}') for i, x in enumerate(parent)]
    result = promote(parent, child, draws=2000, seed=3)
    assert result['support'] > .9  # Paired quality/cost + loose OR allowance.
    assert result['observed_improvement'] is False
    assert result['promoted'] is False


def test_grid_rejects_copied_runs_partial_tasks_and_seed_substitution():
    rows = outcomes('a')
    for bad in (rows+rows, rows[:-1], [replace(r, run_id='copy') for r in rows],
                [replace(r, task=f'{r.task}:seed-{r.repeat}') for r in rows]):
        with pytest.raises(ValueError):
            grid(bad)
    with pytest.raises(ValueError, match='task sets'):
        promote(rows, outcomes('b', tasks=('a', 'b', 'different')), draws=100)
    with pytest.raises(ValueError, match='shared'):
        promote(rows, rows, draws=100)


@pytest.mark.parametrize('field,value', [('tokens', float('nan')), ('quality', 2), ('tokens', -1), ('seconds', True)])
def test_invalid_measurements_fail_closed(field, value):
    with pytest.raises(ValueError):
        replace(outcomes('a')[0], **{field: value})


def test_frontier_keeps_tradeoffs_and_selection_checks_each_domain_and_false_completion():
    archive = {'good': summarize(outcomes('a', 1., 100.)),
               'cheap': summarize(outcomes('b', .8, 50.)),
               'dominated': summarize(outcomes('c', .7, 120.))}
    assert frontier(archive) == ['cheap', 'good']
    assert 'dominated' in archive
    assert select(archive) == 'cheap'
    archive['cheap']['domains']['software'] = .1
    assert select(archive) == 'good'
    archive['good']['false_accept'] = .01
    # A dominated point can still satisfy additional deployment constraints.
    assert select(archive) == 'dominated'
    assert select(archive, quality_floor=.8) is None


def plan():
    value = json.loads(Path('experiments/pareto_pilot.json').read_text())
    value['promotion']['draws'] = 100
    return SearchPlan.model_validate(value)


def report(stage, knobs, run_id, tokens=1000, success=True):
    from src.eval.quality_cases import cases
    from src.eval.quality_transfer_cases import transfer_cases
    domains = {c.name: c.domain for c in cases()+transfer_cases()}
    return {'schema_version': 2, 'run_id': run_id, 'source_id': 'source-v1',
            'evaluation_protocol': 'oracle-v1', 'configuration': dict(knobs, max_cost_usd=.02),
            'model': plan().model, 'provider_only': plan().provider, 'reasoning': 'low',
            'cases_sha256': '|'.join(stage.cases), 'dependency_lock_sha256': 'lock',
            'seed': stage.seed, 'expected_cases': stage.cases,
            'source_unchanged_during_run': True,
            'scenario': 'parameterized_quality_ladder_not_public_benchmark',
            'reports': [{'case': c, 'domain': domains[c], 'seconds': 10., 'passed': success,
                         'answer_passed': success, 'process_passed': success, 'false_accept': False,
                         'independent_check': {'ok': success},
                         'result': {'status': 'done' if success else 'paused', 'tokens': tokens,
                                    'cost': .001, 'cost_estimated': False}} for c in stage.cases]}


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(schema_version=1),
    lambda r: r.update(source_unchanged_during_run=False),
    lambda r: r['reports'].pop(),
    lambda r: r['reports'][0].update(passed=False),
    lambda r: r['reports'][0]['result'].update(cost_estimated=True),
    lambda r: r['reports'][0]['result'].update(cost=float('nan')),
    lambda r: r['reports'][0]['result'].update(tokens=0),
])
def test_bad_reports_cannot_promote(mutation):
    r = report(plan().screen, {}, 'run')
    mutation(r)
    with pytest.raises(ValueError):
        observations(r, 0)


def test_stage_identity_rejects_mixed_versions_copies_and_permission_changes(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path/f'{i}.json'
        save_json(p, report(plan().screen, {}, str(i)))
        paths.append(p)
    first = load_stage(paths)
    second = deepcopy(first)
    second['identity']['configuration']['allow_web'] = True
    with pytest.raises(ValueError, match='Non-tunable'):
        compatible(first, second)
    second['identity']['model'] = 'different'
    with pytest.raises(ValueError, match='model'):
        compatible(first, second)
    broken = json.loads(paths[-1].read_text()); broken['source_id'] = 'new-source'
    save_json(paths[-1], broken)
    with pytest.raises(ValueError, match='variant or tasks'):
        load_stage(paths)
    save_json(paths[-1], json.loads(paths[0].read_text()))
    with pytest.raises(ValueError, match='Duplicate execution'):
        load_stage(paths)


def test_legacy_inventory_never_claims_old_logs_prove_improvement(tmp_path):
    old = report(plan().screen, {}, 'old'); old.pop('schema_version')
    save_json(tmp_path/'old.json', old)
    current = report(plan().screen, {}, 'current')
    save_json(tmp_path/'current.json', current)
    save_json(tmp_path/'copy.json', current)
    result = inventory(tmp_path)
    assert len(result['excluded']) == 1
    assert len(result['groups']) == 1
    assert next(iter(result['groups'].values()))['repeats'] == 1
    assert not next(iter(result['groups'].values()))['ready_for_comparison']


def test_plan_family_split_budget_and_tunable_boundary():
    p = plan()
    assert p.envelope()['maximum_task_executions'] == 63
    assert p.envelope()['maximum_reserved_usd'] == 1.26
    assert p.configurations()['prefetch_completion']['source_prefetch_bytes'] == 2400
    bad = p.model_dump(); bad['test'] = bad['screen']
    with pytest.raises(ValueError, match='disjoint'):
        SearchPlan.model_validate(bad)
    bad = p.model_dump(); bad['variants'][1]['knobs']['allow_web'] = True
    with pytest.raises(ValueError):
        SearchPlan.model_validate(bad)


def test_campaign_retains_dominated_parent_and_freezes_selection_before_test(tmp_path):
    p = plan()
    seen = []
    async def evaluate(plan, stage, knobs, target, work):
        name = target.parent.parent.name
        # The intermediate profile improves screening but is dominated on dev.
        tokens = {'baseline': 1000, 'prefetch': 800, 'prefetch_completion': 600}[name]
        if name == 'prefetch' and stage == p.dev:
            tokens = 1200
        if stage == p.test:
            saved = json.loads((tmp_path/'search/archive.json').read_text())
            assert saved['selected'] == 'prefetch_completion'
            assert saved['status'] == 'selected_on_dev'
        seen.append((name, stage.seed))
        save_json(target, report(stage, knobs, str(target), tokens))
    result = asyncio.run(campaign(p, tmp_path/'search', evaluate, lambda: 'stable'))
    assert result['status'] == 'validated_on_reserved_families'
    assert result['variants']['prefetch']['status'] == 'retained'
    assert result['frontier'] == ['prefetch_completion']
    assert result['selected'] == 'prefetch_completion'
    assert result['reserved_usd'] == 1.26
    assert len(seen) == 21  # 3 repeats x (6 candidate stages + 1 test).
    profile = json.loads((tmp_path/'search/selected-profile.json').read_text())
    assert profile['eligible_for_manual_adoption'] is True
    assert result['production_changed'] is False
    with pytest.raises(FileExistsError):
        asyncio.run(campaign(p, tmp_path/'search', evaluate, lambda: 'stable'))


def test_campaign_test_failure_cannot_select_a_different_winner(tmp_path):
    p = plan()
    async def evaluate(plan, stage, knobs, target, work):
        tokens = 500 if knobs['source_prefetch_bytes'] else 1000
        if knobs['completion_review']:
            tokens = 300
        save_json(target, report(stage, knobs, str(target), tokens, success=stage != p.test))
    result = asyncio.run(campaign(p, tmp_path/'search', evaluate, lambda: 'stable'))
    assert result['status'] == 'test_failed_no_deployment'
    assert result['selected'] == 'prefetch_completion'
    assert not json.loads((tmp_path/'search/selected-profile.json').read_text())['eligible_for_manual_adoption']


def test_campaign_budget_reserved_before_call_and_partial_evidence_retained(tmp_path):
    value = plan().model_dump(); value['total_budget_usd'] = .06
    p = SearchPlan.model_validate(value)
    calls = []
    async def evaluate(plan, stage, knobs, target, work):
        saved = json.loads((tmp_path/'search/archive.json').read_text())
        assert saved['reserved_usd'] == .06
        calls.append(1)
        save_json(target, report(stage, knobs, str(target)))
    with pytest.raises(ValueError, match='budget exhausted'):
        asyncio.run(campaign(p, tmp_path/'search', evaluate, lambda: 'stable'))
    saved = json.loads((tmp_path/'search/archive.json').read_text())
    assert len(calls) == 1 and saved['status'] == 'stopped'
    assert saved['reservations'][0]['state'] == 'completed'


def test_campaign_source_change_stops_before_any_further_calls(tmp_path):
    marker = ['v1']
    async def evaluate(plan, stage, knobs, target, work):
        save_json(target, report(stage, knobs, str(target)))
        marker[0] = 'v2'
    with pytest.raises(ValueError, match='changed during evaluation'):
        asyncio.run(campaign(plan(), tmp_path/'search', evaluate, lambda: marker[0]))
    saved = json.loads((tmp_path/'search/archive.json').read_text())
    assert saved['reserved_usd'] == .06
    assert len(saved['reservations']) == 1


def test_campaign_rejection_skips_child_and_does_not_buy_dev_evaluation(tmp_path):
    p = plan()
    async def evaluate(plan, stage, knobs, target, work):
        # Cheaper only because the candidate fails every requirement.
        success = not knobs['source_prefetch_bytes']
        save_json(target, report(stage, knobs, str(target), 1000 if success else 1, success))
    result = asyncio.run(campaign(p, tmp_path/'search', evaluate, lambda: 'stable'))
    assert result['variants']['prefetch']['status'] == 'rejected_at_screen'
    assert 'dev' not in result['variants']['prefetch']
    assert result['variants']['prefetch_completion']['status'] == 'parent_not_promoted'
    assert result['selected'] == 'baseline'
    assert result['reserved_usd'] == .72


def test_real_quality_runner_emits_ingestible_provenance(tmp_path, monkeypatch):
    """Exercise the actual reporter; the actor is a fixture, not quality evidence."""
    from types import SimpleNamespace
    from scripts import quality_ladder
    import src.harness
    case = quality_ladder.cases(42)[0]
    class ActorFixture:
        def __init__(self, *args, **kwargs):
            self.control = SimpleNamespace(orchestrator=None)
            self.supervisor = SimpleNamespace(tool_counts={'main': {'math': 1}})
        async def run(self, *args, **kwargs):
            return {'status': 'done', 'cost': .001, 'cost_estimated': False, 'tokens': 100,
                    'final_answer': json.dumps({'value': str(case.expected), 'derivation': 'fixture'})}
    monkeypatch.setattr(src.harness, 'Harness', ActorFixture)
    args = quality_ladder.build_parser().parse_args([])
    args.seed, args.case = 42, [case.name]
    args.output, args.work_dir = str(tmp_path/'result.json'), str(tmp_path/'work')
    asyncio.run(quality_ladder.run(args))
    saved = json.loads(Path(args.output).read_text())
    rows = observations(saved, 0)
    assert len(rows) == 1 and rows[0].quality == 1
    assert saved['expected_cases'] == [case.name]
    assert saved['source_unchanged_during_run'] is True
    assert len(saved['evaluation_protocol']) == 64
