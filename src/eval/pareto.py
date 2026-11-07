"""Offline quality/token selection inspired by ParetoSaddler (arXiv:2611.02079).

Promotion and deployment are deliberately separate. The bootstrap conditions on
fixed tasks: its support is NOT a calibrated probability of generalization.
No model calls, runtime changes, or third-party optimization dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean


@dataclass(frozen=True)
class Observation:
    task: str
    family: str
    domain: str
    repeat: int
    run_id: str
    quality: float
    tokens: float
    seconds: float
    false_accept: bool = False

    def __post_init__(self):
        for name in ('task', 'family', 'domain', 'run_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'Missing {name}')
        if type(self.repeat) is not int or self.repeat < 0:
            raise ValueError('Invalid repeat')
        for name in ('quality', 'tokens', 'seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {name}')
        if self.quality > 1 or type(self.false_accept) is not bool:
            raise ValueError('Invalid outcome')
        if self.false_accept and self.quality:
            raise ValueError('False completion cannot be a successful outcome')


def grid(rows, minimum_repeats=3):
    """Reject duplicates/partial grids; task seeds are tasks, not repetitions."""
    if not rows:
        raise ValueError('No observations')
    grouped, ids = {}, set()
    for row in rows:
        if row.run_id in ids:
            raise ValueError('Duplicate execution; copied reports are not repetitions')
        ids.add(row.run_id)
        grouped.setdefault(row.task, []).append(row)
    expected = None
    for task, values in grouped.items():
        values.sort(key=lambda x: x.repeat)
        repeats = [x.repeat for x in values]
        if len(set(repeats)) != len(repeats):
            raise ValueError(f'Duplicate repeat: {task}')
        if len(repeats) < minimum_repeats:
            raise ValueError(f'Insufficient repetitions: {task}')
        if len({(x.family, x.domain) for x in values}) != 1:
            raise ValueError('Task metadata changed')
        if expected is not None and repeats != expected:
            raise ValueError('Incomplete repetition grid')
        expected = repeats
    return grouped


def summarize(rows):
    grouped = grid(rows)
    task_means = [{
        'domain': values[0].domain,
        'quality': mean(x.quality for x in values),
        'tokens': mean(x.tokens for x in values),
        'seconds': mean(x.seconds for x in values),
        'false_accept': mean(x.false_accept for x in values),
    } for values in grouped.values()]
    result = {key: mean(t[key] for t in task_means)
              for key in ('quality', 'tokens', 'seconds', 'false_accept')}
    result.update(tasks=len(grouped), runs=len(rows), cost_unit='total_tokens',
                  domains={d: mean(t['quality'] for t in task_means if t['domain'] == d)
                           for d in sorted({t['domain'] for t in task_means})})
    result['reliability']=reliability(rows)
    return result


def reliability(rows):
    """Task-balanced observed k-subset success, never mean(pass@1)**k.

    The combinatorial estimators require exchangeable trials to predict future
    repeats. Here they describe the fixed task grid, not population guarantees.
    pass@k is oracle coverage, not a working mechanism for selecting an answer.
    """
    grouped=grid(rows)
    if any(x.quality not in (0.,1.) for x in rows):
        return {'available':False,'reason':'binary independently verified outcomes required'}
    n=len(next(iter(grouped.values())))
    counts=[sum(int(x.quality) for x in values) for values in grouped.values()]
    choose=lambda a,k:math.comb(a,k) if a>=k else 0
    return {'available':True,'repeats':n,'tasks':len(grouped),
        'pass_at_k':{str(k):mean(1-choose(n-c,k)/choose(n,k) for c in counts) for k in range(1,n+1)},
        'pass_power_k':{str(k):mean(choose(c,k)/choose(n,k) for c in counts) for k in range(1,n+1)},
        'all_repeats_passed':mean(c==n for c in counts),
        'unstable_tasks':sorted(task for task,values in grouped.items() if 0<sum(x.quality for x in values)<n),
        'false_accept_in_any_repeat':mean(any(x.false_accept for x in values) for values in grouped.values()),
        'scope':'observed fixed-task subsets; no generalization guarantee; pass@k requires an oracle selector'}


def promote(parent, candidate, *, draws=10000, threshold=.7,
            quality_loss=.1, cost_growth=.5, seed=0):
    """Dirichlet(1) weights shared by quality/cost within each execution.

    Parent and candidate weights are independent, tasks are equally weighted.
    Unlike the paper's unbounded epsilon_C, our pilot allows at most 50% token
    growth during exploration. Deployment uses its own explicit constraints.
    """
    if type(draws) is not int or not 100 <= draws <= 100000:
        raise ValueError('draws must be in [100, 100000]')
    for name, value in [('threshold', threshold), ('quality_loss', quality_loss), ('cost_growth', cost_growth)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid {name}')
    if not .5 < threshold <= 1 or not 0 <= quality_loss <= 1 or cost_growth < 0:
        raise ValueError('Invalid promotion allowances')
    left, right = grid(parent), grid(candidate)
    if set(left) != set(right):
        raise ValueError('Parent/candidate task sets differ')
    if {x.run_id for x in parent} & {x.run_id for x in candidate}:
        raise ValueError('Executions cannot be shared by different candidates')
    for task in left:
        if [(x.repeat, x.family, x.domain) for x in left[task]] != [(x.repeat, x.family, x.domain) for x in right[task]]:
            raise ValueError('Parent/candidate repetition protocols differ')
    rng = random.Random(seed)

    def sample(group):
        q = c = 0.
        for task in sorted(group):
            rows = group[task]
            weights = [rng.expovariate(1.) for _ in rows]
            total = sum(weights)
            q += sum(w*x.quality for w, x in zip(weights, rows))/total
            c += sum(w*x.tokens for w, x in zip(weights, rows))/total
        return q/len(group), c/len(group)

    supported = 0
    for _ in range(draws):
        pq, pc = sample(left)
        cq, cc = sample(right)
        # Multiplication handles zero-cost parents without division by zero:
        # a free parent cannot admit positive cost under a finite growth bound.
        better_quality = cq > pq + 1e-12 and cc <= pc*(1+cost_growth) + 1e-12
        cheaper = cc < pc - 1e-12 and cq >= pq-quality_loss-1e-12
        supported += better_quality or cheaper
    support = supported/draws
    parent_summary, candidate_summary = summarize(parent), summarize(candidate)
    # With loose loss allowances, the OR-event can have high bootstrap support
    # even for identical empirical distributions. Do not retain a no-op or a
    # candidate empirically worse in both objectives just because of that gate.
    observed_improvement = (candidate_summary['quality'] > parent_summary['quality']+1e-12
                            or candidate_summary['tokens'] < parent_summary['tokens']-1e-12)
    return {'promoted': observed_improvement and support >= threshold, 'support': support,
            'observed_improvement': observed_improvement,
            'threshold': threshold, 'draws': draws, 'seed': seed,
            'quality_loss': quality_loss, 'cost_growth': cost_growth,
            'parent': parent_summary, 'candidate': candidate_summary,
            'uncertainty_scope': 'execution variation on these fixed tasks only'}


def frontier(summaries):
    """Empirical frontier; dominated candidates remain in the caller's archive."""
    return sorted(name for name, a in summaries.items() if not any(
        b['quality'] >= a['quality'] and b['tokens'] <= a['tokens'] and
        (b['quality'] > a['quality'] or b['tokens'] < a['tokens'])
        for other, b in summaries.items() if other != name))


def select(summaries, *, quality_floor=.7, domain_floor=.5,
           max_tokens=80000, max_seconds=300, max_false_accept=0., consistency_floor=None):
    """Deployment selection uses dev data, never screening/test scores."""
    bounds = (quality_floor, domain_floor, max_tokens, max_seconds, max_false_accept)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds):
        raise ValueError('Selection bounds must be finite numbers')
    if not 0 <= quality_floor <= 1 or not 0 <= domain_floor <= 1 or not 0 <= max_false_accept <= 1 or min(max_tokens, max_seconds) <= 0:
        raise ValueError('Invalid selection bounds')
    if consistency_floor is not None and (isinstance(consistency_floor,bool) or not isinstance(consistency_floor,(int,float))
            or not math.isfinite(consistency_floor) or not 0<=consistency_floor<=1):
        raise ValueError('Invalid consistency floor')
    admissible = [name for name, s in summaries.items()
                  if s['quality'] >= quality_floor and s['tokens'] <= max_tokens
                  and s['seconds'] <= max_seconds and s['false_accept'] <= max_false_accept
                  and s['domains'] and all(q >= domain_floor for q in s['domains'].values())
                  and (consistency_floor is None or (s.get('reliability',{}).get('available')
                       and s['reliability']['all_repeats_passed']>=consistency_floor))]
    return min(admissible, key=lambda name: (summaries[name]['tokens'], -summaries[name]['quality'], name)) if admissible else None
