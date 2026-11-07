"""Provenance checks for offline harness comparison and legacy-log inventory."""
from __future__ import annotations

from dataclasses import asdict
import asyncio
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .pareto import Observation, frontier, grid, promote, select, summarize


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                   ensure_ascii=False).encode()).hexdigest()


def read_report(path):
    path = Path(path)
    if path.stat().st_size > 32*1024*1024:
        raise ValueError('Oversized evaluation report')
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get('scenario') != 'parameterized_quality_ladder_not_public_benchmark':
        raise ValueError('Not a quality-ladder report')
    return value


def observations(report, repeat):
    """Only complete, attributable, unmodified executions may guide selection."""
    if report.get('schema_version') != 2:
        raise ValueError('Legacy report lacks execution/protocol provenance')
    required = ('run_id', 'source_id', 'evaluation_protocol', 'configuration',
                'cases_sha256', 'dependency_lock_sha256', 'expected_cases')
    if any(not report.get(k) for k in required):
        raise ValueError('Missing provenance')
    if report.get('source_unchanged_during_run') is not True:
        raise ValueError('Source changed or stability not recorded')
    expected = report['expected_cases']
    rows = report.get('reports', [])
    if len(set(expected)) != len(expected) or sorted(x['case'] for x in rows) != sorted(expected):
        raise ValueError('Incomplete or duplicated tasks')
    output = []
    for row in rows:
        result = row['result']
        flags = ('passed', 'answer_passed', 'process_passed', 'false_accept')
        if any(type(row.get(k)) is not bool for k in flags):
            raise ValueError('Missing independent outcome flags')
        if row.get('independent_check', {}).get('ok') is not row['answer_passed']:
            raise ValueError('Independent check and score disagree')
        success = row['answer_passed'] and row['process_passed'] and result.get('status') == 'done'
        false_accept = result.get('status') == 'done' and not row['answer_passed']
        if row['passed'] != success or row['false_accept'] != false_accept:
            raise ValueError('Outcome and completion disagree')
        if (result.get('status') == 'error' or result.get('cost_estimated') is not False
                or result.get('token_budget', {}).get('estimated', 0)
                or result.get('failed_requests', 0) or result.get('tokens', 0) <= 0):
            raise ValueError('Uncertain usage or infrastructure failure; retain report, stop comparison')
        if (isinstance(result.get('cost'), bool) or not isinstance(result.get('cost'), (int, float))
                or not math.isfinite(result['cost']) or result['cost'] < 0):
            raise ValueError('Missing or invalid reported USD cost')
        output.append(Observation(
            task=f"{row['case']}:{report['seed']}", family=row['case'], domain=row['domain'],
            repeat=repeat, run_id=f"{report['run_id']}:{row['case']}",
            quality=float(success), tokens=result['tokens'], seconds=row['seconds'],
            false_accept=false_accept))
    return output


def identity(report):
    return {k: report.get(k) for k in ('source_id', 'configuration', 'model',
            'provider_only', 'reasoning', 'evaluation_protocol', 'dependency_lock_sha256')}


def load_stage(paths):
    """A repeated stage uses one exact task set and one immutable variant."""
    records, rows, expected, protocol = [], [], None, None
    for repeat, path in enumerate(paths):
        report = read_report(path)
        key = digest(identity(report))
        task_set = (report.get('seed'), report.get('cases_sha256'), sorted(report.get('expected_cases', [])))
        if protocol is not None and (key != protocol or task_set != expected):
            raise ValueError('Repeated executions changed variant or tasks')
        protocol, expected = key, task_set
        rows.extend(observations(report, repeat))
        records.append({'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    grid(rows)
    return {'identity': identity(report), 'tasks': expected, 'records': records,
            'observations': [asdict(x) for x in rows], 'summary': summarize(rows)}


def compatible(parent, candidate):
    """Candidate knobs may differ; model/evaluator/source/task protocol may not."""
    left, right = parent['identity'], candidate['identity']
    for key in ('model', 'provider_only', 'reasoning', 'evaluation_protocol', 'dependency_lock_sha256', 'source_id'):
        if left[key] != right[key]:
            raise ValueError(f'Incompatible comparison: {key}')
    if parent['tasks'] != candidate['tasks']:
        raise ValueError('Incompatible task set')
    tunable = set(Knobs.model_fields)
    if ({k: v for k, v in left['configuration'].items() if k not in tunable} !=
            {k: v for k, v in right['configuration'].items() if k not in tunable}):
        raise ValueError('Non-tunable configuration changed (budget, permissions, backend or checks)')


def inventory(directory):
    """Legacy evidence remains diagnostic: never infer repetitions from seeds."""
    groups, rejected = {}, []
    for path in sorted(Path(directory).glob('*.json')):
        try:
            report = read_report(path)
        except (ValueError, OSError):
            continue
        try:
            rows = observations(report, 0)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({'path': str(path), 'reason': str(exc)})
            continue
        key = digest({'variant': identity(report), 'tasks': report['cases_sha256'], 'seed': report['seed']})
        group = groups.setdefault(key, {'paths': [], 'run_ids': [], 'tasks': len(rows)})
        if report['run_id'] not in group['run_ids']:
            group['run_ids'].append(report['run_id']); group['paths'].append(str(path))
    for group in groups.values():
        group['repeats'] = len(group['run_ids'])
        group['ready_for_comparison'] = group['repeats'] >= 3
    return {'groups': groups, 'excluded': rejected,
            'note': 'No promotion from single runs, mixed sources, or legacy reports; no API calls.'}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Knobs(StrictModel):
    context_policy: Literal['bounded', 'single_action'] | None = None
    planning: Literal['adaptive', 'required'] | None = None
    source_prefetch_bytes: int | None = Field(default=None, ge=0, le=4800)
    result_memory: bool | None = None
    completion_review: bool | None = None
    adaptive_tools: bool | None = None


class Variant(StrictModel):
    name: str = Field(pattern=r'^[a-z][a-z0-9_-]{0,39}$')
    parent: str | None = None
    knobs: Knobs = Field(default_factory=Knobs)


class Stage(StrictModel):
    seed: int = Field(ge=0, le=2**31-1)
    cases: list[str] = Field(min_length=1, max_length=10)


class Promotion(StrictModel):
    draws: int = Field(default=10000, ge=100, le=100000)
    threshold: float = Field(default=.7, gt=.5, le=1)
    quality_loss: float = Field(default=.1, ge=0, le=1)
    cost_growth: float = Field(default=.5, ge=0, le=2)
    seed: int = Field(default=0, ge=0)


class Deployment(StrictModel):
    consistency_floor: float | None = Field(default=None, ge=0, le=1)
    quality_floor: float = Field(default=.7, ge=0, le=1)
    domain_floor: float = Field(default=.5, ge=0, le=1)
    max_tokens: int = Field(default=80000, gt=0, le=80000)
    max_seconds: float = Field(default=180., gt=0, le=300)
    max_false_accept: float = Field(default=0., ge=0, le=1)


class SearchPlan(StrictModel):
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    reasoning: Literal['low', 'medium', 'high'] = 'low'
    repeats: int = Field(default=3, ge=3, le=5)
    budget_per_task_usd: float = Field(default=.02, gt=0, le=.1)
    total_budget_usd: float = Field(default=1.3, gt=0, le=5)
    timeout_seconds: int = Field(default=180, ge=10, le=300)
    max_campaign_seconds: int = Field(default=1800, ge=60, le=7200)
    variants: list[Variant] = Field(min_length=2, max_length=6)
    screen: Stage
    dev: Stage
    test: Stage
    promotion: Promotion = Field(default_factory=Promotion)
    deployment: Deployment = Field(default_factory=Deployment)

    @model_validator(mode='after')
    def consistent(self):
        from .quality_cases import cases
        from .quality_transfer_cases import transfer_cases
        catalog = {c.name: c.domain for c in cases()+transfer_cases()}
        seen = set()
        for stage in (self.screen, self.dev, self.test):
            if len(set(stage.cases)) != len(stage.cases) or any(c not in catalog for c in stage.cases):
                raise ValueError('Unknown or duplicate case')
            if seen.intersection(stage.cases):
                raise ValueError('Screen/dev/test must use disjoint task families, not just new seeds')
            seen.update(stage.cases)
            if {catalog[c] for c in stage.cases} != {'software', 'mathematics', 'algorithms'}:
                raise ValueError('Every stage must cover software, mathematics and algorithms')
        names = set()
        for i, variant in enumerate(self.variants):
            if variant.name in names or (i == 0 and variant.parent is not None):
                raise ValueError('Invalid baseline or duplicate variant')
            if i and variant.parent not in names:
                raise ValueError('Parent must be an earlier variant')
            names.add(variant.name)
        return self

    def envelope(self):
        # All candidates promoted is the maximum; one test winner only.
        tasks = self.repeats*(len(self.variants)*(len(self.screen.cases)+len(self.dev.cases))+len(self.test.cases))
        return {'maximum_task_executions': tasks,
                'maximum_reserved_usd': round(tasks*self.budget_per_task_usd, 8),
                'authorized_campaign_cap_usd': self.total_budget_usd,
                'campaign_wall_time_limit_seconds': self.max_campaign_seconds,
                'fits_full_search': tasks*self.budget_per_task_usd <= self.total_budget_usd+1e-10,
                'cost_objective': 'total_tokens; USD and latency are separately recorded',
                'calls_without_live': 0}

    def configurations(self):
        configs = {}
        for variant in self.variants:
            configs[variant.name] = dict(configs.get(variant.parent, {
                'planning': 'adaptive', 'source_prefetch_bytes': 0, 'context_policy': 'bounded',
                'result_memory': True, 'completion_review': False, 'adaptive_tools': True}))
            configs[variant.name].update(variant.knobs.model_dump(exclude_none=True))
        if len({digest(c) for c in configs.values()}) != len(configs):
            raise ValueError('Duplicate effective configurations')
        return configs


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


async def campaign(plan, directory, evaluate, fingerprint):
    """Bounded configuration search. evaluate is the real quality-ladder runner.

    Reserve full per-task allowances BEFORE each group. Interrupted groups keep
    their reservations/evidence; rerunning never silently overwrites a campaign.
    Promoted-but-dominated variants remain eligible as parents. No auto-deploy.
    """
    configs = plan.configurations()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    initial = fingerprint()
    deadline = time.monotonic()+plan.max_campaign_seconds
    archive = {'version': 1, 'plan': plan.model_dump(), 'plan_id': digest(plan.model_dump()),
               'fingerprint': initial, 'status': 'running', 'reserved_usd': 0.,
               'reservations': [], 'variants': {}, 'frontier': [], 'selected': None,
               'test_used_for_selection': False, 'production_changed': False}

    def persist():
        save_json(directory/'archive.json', archive)

    async def collect(name, phase):
        stage = getattr(plan, phase)
        record = archive['variants'][name]
        stage_dir = directory/name/phase
        stage_dir.mkdir(parents=True)
        paths = []
        for repeat in range(plan.repeats):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise ValueError('Campaign wall-time budget exhausted')
            if fingerprint() != initial:
                raise ValueError('Source/config changed during campaign')
            reserve = len(stage.cases)*plan.budget_per_task_usd
            if archive['reserved_usd']+reserve > plan.total_budget_usd+1e-10:
                raise ValueError('Campaign budget exhausted before next evaluation')
            target = stage_dir/f'repeat-{repeat}.json'
            archive['reserved_usd'] = round(archive['reserved_usd']+reserve, 8)
            receipt = {'variant': name, 'phase': phase, 'repeat': repeat,
                       'reserved_usd': reserve, 'path': str(target), 'state': 'started'}
            archive['reservations'].append(receipt)
            persist()
            await asyncio.wait_for(
                evaluate(plan, stage, configs[name], target, stage_dir/f'work-{repeat}'),
                timeout=remaining)
            if fingerprint() != initial:
                raise ValueError('Source/config changed during evaluation')
            report = read_report(target)
            observations(report, repeat)
            if (sorted(report['expected_cases']) != sorted(stage.cases) or report['seed'] != stage.seed
                    or report['model'] != plan.model or report['provider_only'] != plan.provider
                    or report['reasoning'] != plan.reasoning):
                raise ValueError('Runner did not execute the requested protocol')
            if any(report['configuration'].get(k) != v for k, v in configs[name].items()):
                raise ValueError('Runner did not execute the requested knobs')
            actual = sum(row['result']['cost'] for row in report['reports'])
            if actual < 0 or actual > reserve+1e-8:
                raise ValueError('Reported cost exceeded reserved envelope')
            receipt.update(state='completed', reported_cost_usd=actual)
            paths.append(target)
            persist()
        data = load_stage(paths)
        # All phases of a variant must use the same source/configuration.
        for other in ('screen', 'dev', 'test'):
            if other in record and data['identity'] != record[other]['identity']:
                raise ValueError('Variant changed between evaluation stages')
        record[phase] = data
        persist()
        return data

    persist()
    try:
        baseline = plan.variants[0].name
        for variant in plan.variants:
            record = archive['variants'][variant.name] = {
                'parent': variant.parent, 'knobs': configs[variant.name], 'status': 'pending'}
            if variant.parent and archive['variants'][variant.parent]['status'] != 'retained':
                record['status'] = 'parent_not_promoted'
                persist()
                continue
            screen = await collect(variant.name, 'screen')
            if variant.parent:
                parent = archive['variants'][variant.parent]['screen']
                compatible(parent, screen)
                record['promotion'] = promote(
                    [Observation(**x) for x in parent['observations']],
                    [Observation(**x) for x in screen['observations']], **plan.promotion.model_dump())
                if not record['promotion']['promoted']:
                    record['status'] = 'rejected_at_screen'
                    persist()
                    continue
            dev = await collect(variant.name, 'dev')
            if variant.name != baseline:
                compatible(archive['variants'][baseline]['dev'], dev)
            record['status'] = 'retained'
            summaries = {n: v['dev']['summary'] for n, v in archive['variants'].items() if v['status'] == 'retained'}
            archive['frontier'] = frontier(summaries)
            persist()
        summaries = {n: v['dev']['summary'] for n, v in archive['variants'].items() if v['status'] == 'retained'}
        winner = select(summaries, **plan.deployment.model_dump())
        archive['selected'] = winner
        archive['status'] = 'no_admissible_candidate' if winner is None else 'selected_on_dev'
        persist()  # Selection frozen before exposing any test result.
        if winner is not None:
            test = await collect(winner, 'test')
            passed = select({winner: test['summary']}, **plan.deployment.model_dump()) is not None
            archive['status'] = 'validated_on_reserved_families' if passed else 'test_failed_no_deployment'
            # The exact profile is a reviewable output, never installed by this runner.
            save_json(directory/'selected-profile.json', {
                'candidate': winner, 'model': plan.model, 'provider': plan.provider,
                'reasoning': plan.reasoning, 'coding_overrides': configs[winner],
                'evaluated_configuration': test['identity']['configuration'],
                'evaluation_protocol': test['identity']['evaluation_protocol'],
                'eligible_for_manual_adoption': passed, 'plan_id': archive['plan_id'],
                'source_id': test['identity']['source_id'],
                'note': 'Diagnostic family split; not a public benchmark. Transfer requires the evaluated configuration, not just the tuned knobs; task-specific checks belong to the evaluation protocol.'})
        persist()
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
        archive.update(status='stopped', error=type(exc).__name__+': '+str(exc)[:1000])
        persist()
        raise
    return archive
