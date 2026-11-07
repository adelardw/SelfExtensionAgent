"""Turn acceptance gaps into a small, revision-aware work queue, without inference."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class NextStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    requirement: str = Field(pattern=r'^r[0-9]+$')
    problem: str = Field(min_length=1, max_length=400)
    action: str = Field(min_length=1, max_length=500)
    mode: Literal['inspect', 'reason', 'research', 'repair', 'implement', 'verify']
    verification: str = Field(min_length=1, max_length=400)


def agenda(contract, graph, missing, *, mode, revision, steering_version):
    """Concrete work before final checks and catch-all gaps; retain provenance.

    No queue item can grant authority or mark a requirement supported. Only a new
    observation/review can close it. Unreviewed semantic requirements aren't failures.
    """
    if contract is None:
        return {}
    report = graph.packet()
    scope = graph.data.get('followthrough_scope', {})
    same_task = scope.get('contract_version') == graph.data['version']
    stale = (scope.get('revision') != revision or scope.get('steering_version') != steering_version)
    advice = {s['requirement']: s for s in graph.data.get('next_steps', [])} if same_task else {}
    statuses = {r['id']: r['status'] for r in report['requirements']}
    queue = []
    for rule in sorted(missing, key=lambda r:r.validation=='registered_checks'):
        if rule.validation == 'artifact_exists':
            paths = ', '.join(dict.fromkeys(p for p in [rule.path, *rule.paths] if p))
            action = f'Create or repair {paths}; then inspect the actual delivered files.'
            verify = f'Files {paths} exist in the delivery workspace and satisfy: {rule.text}'
            selected = 'implement'
        elif rule.validation == 'registered_checks':
            action, verify, selected = 'Run the relevant registered check after the last edit.', 'Current revision passes the relevant check.', 'verify'
        elif rule.validation == 'new_or_updated_tests':
            action, verify, selected = 'Add the requested behavioral tests, then run them.', 'Tests exercise the requested behavior, not only the environment.', 'implement'
        else:
            action, verify, selected = 'Inspect and integrate the requested worker changes.', 'Parent checks pass after integration.', 'implement'
        queue.append({'requirement': rule.id, 'problem': rule.text, 'action': action,
                      'verification': verify, 'mode': selected, 'status': 'observed_missing'})
    present = {s['requirement'] for s in queue}
    for rule in contract.requirements:
        if rule.id in present or statuses.get(rule.id) == 'supported':
            continue
        if rule.id in advice:
            step={**advice[rule.id], 'status': 'recheck_after_change' if stale else 'review_gap'}
            if stale:
                step.update(mode='inspect',action='Source changed since this review. Check whether the reported gap remains before repeating its repair: '+step['problem'])
            queue.append(step)
        elif statuses.get(rule.id) in {'missing', 'uncertain', 'stale'}:
            queue.append({'requirement': rule.id, 'problem': rule.text,
                'action': 'Inspect the specific unresolved criterion and obtain a discriminating observation.',
                'mode': 'inspect', 'verification': 'Evidence addresses this criterion on the current revision.',
                'status': 'recheck_after_change' if stale else 'review_gap'})
    # A generic "complete the product" criterion must not switch repair back to
    # inspect ahead of the critic's specific fix. Checks follow known unfinished
    # work instead of running once after every file edit. All gates remain present.
    queue.sort(key=lambda step: 0 if step['status']=='observed_missing' and step['mode']!='verify'
               else 1 if step['requirement'] in advice
               else 2 if step['status']=='observed_missing' else 3)
    focus = queue[0] if queue else None
    return {'remaining': len(queue), 'unreviewed':sum(r['status']=='unreviewed' for r in report['requirements']),
            'items':[{k:s[k] for k in ('requirement','mode','status')} | {'problem':s['problem'][:180]} for s in queue[:6]], 'focus': focus,
            'current_mode': mode, 'suggested_mode': focus['mode'] if focus else mode,
            'instruction': ('Complete one concrete gap, obtain the stated verification, then move on. '
                'After a source change recheck old findings before applying the same repair again. '
                'A missing dependency is a hypothesis: test its import/availability before installing or replacing it. '
                'Ask the user only for a material requirement/choice absent from task, steering and evidence; '
                'ordinary implementation details and test failures are yours to resolve.')}
