"""Shared request admission across actors, critics, routers and retrieval.

Reservations use conservative input byte bounds and the actual output cap. They
are not provider tokenization guarantees; unexpected usage is recorded, never
clipped to make a run appear to respect its limit.
"""
from contextlib import contextmanager
from contextvars import ContextVar


class BudgetExceeded(ValueError):
    """Pause before admitting another request beyond the configured envelope."""


active_budget = ContextVar('coding_token_budget', default=None)


class TokenBudget:
    def __init__(self, limit):
        self.limit = int(limit)
        if self.limit <= 0:
            raise ValueError('max_total_tokens must be positive')
        self.spent = self.reserved = self.estimated = self.overruns = 0
        self.engines = set()

    def attach(self, engine):
        if engine not in self.engines:
            self.engines.add(engine)
            # Continuations may hydrate model counters before entering the run.
            self.spent += max(0, int(engine.usage().get('tokens', 0)))

    def reserve(self, amount):
        amount = int(amount)
        if amount < 0:
            raise ValueError('Token reservation cannot be negative')
        # No await between admission and reservation: parallel asyncio callers
        # cannot each spend the same remaining capacity.
        if self.spent + self.reserved + amount > self.limit:
            raise BudgetExceeded(
                f'Shared token budget unavailable: {self.spent} accounted + '
                f'{self.reserved} reserved + {amount} request bound > {self.limit}; checkpoint retained')
        self.reserved += amount
        return TokenReservation(self, amount)

    def output_capacity(self, prompt_bound, requested):
        """Use remaining capacity without expanding either input or output caps."""
        available = self.limit-self.spent-self.reserved-int(prompt_bound)
        # Do not buy a prompt that leaves only a handful of output tokens. Small
        # explicitly bounded calls retain their original smaller allowance.
        minimum = min(int(requested), 512)
        if available < minimum:
            raise BudgetExceeded(
                f'Shared token budget unavailable: {self.spent} accounted + {self.reserved} reserved; '
                f'{prompt_bound} input bound leaves {max(0, available)} output tokens '
                f'(minimum {minimum}) within {self.limit}; checkpoint retained')
        return min(int(requested), available)

    def packet(self):
        return {'limit': self.limit, 'accounted': self.spent, 'reserved': self.reserved,
                'estimated': self.estimated, 'remaining': max(0, self.limit-self.spent-self.reserved),
                'provider_bound_overruns': self.overruns}


class TokenReservation:
    def __init__(self, budget, amount):
        self.budget, self.amount = budget, amount
        self.sent = False
        self.actual = None
        self.closed = False

    def __enter__(self):
        return self

    def usage(self, value, input_key='prompt_tokens', output_key='completion_tokens'):
        counts = [value.get(input_key), value.get(output_key)]
        if all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts):
            self.actual = sum(counts)

    def rejected(self):
        self.actual = 0

    def __exit__(self, *_):
        if self.closed:
            return
        self.closed = True
        charged = self.actual if self.actual is not None else self.amount if self.sent else 0
        self.budget.reserved -= self.amount
        self.budget.spent += charged
        if self.sent and self.actual is None:
            self.budget.estimated += charged
        if charged > self.amount:
            self.budget.overruns += 1


def budget_for(engine):
    budget = active_budget.get()
    if budget is None:
        budget = getattr(engine, '_token_budget', None)
        if budget is None:
            budget = engine._token_budget = TokenBudget(engine.config.get('max_total_tokens', 120000))
    budget.attach(engine)
    return budget


@contextmanager
def token_scope(limit):
    # A nested run cannot reset its parent's allowance.
    budget = active_budget.get() or TokenBudget(limit)
    token = active_budget.set(budget)
    try:
        yield budget
    finally:
        active_budget.reset(token)
