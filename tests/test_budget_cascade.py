"""Каскад ярусов + интерактив-доведение без «продолжи».
- Финальную сборку делает УМНАЯ модель (deep), легвóрк — дёшево.
- Интерактив (desktop/TUI): план доводится до естественного конца, стоп только по hard-предохранителю.
- eval/one-shot: тугой бюджет как раньше.
"""
from unittest.mock import patch

import src.graph.agent as A


def test_synthesis_tier_is_smart_not_fast():
    """Финал графа собирает deep (умная), а НЕ fast (дешёвая) — каскадный reduce."""
    # synth_chain построен из deep_llm (config synthesis_model: deep)
    assert A.synth_chain.steps[-1] is A.deep_llm
    assert A.deep_llm is not A.llm  # точно не дешёвый fast


def test_research_synthesis_tier_smart():
    import src.data.research as R
    assert R._SYNTH_ROLE == "deep"  # research-синтез тоже умной моделью


def test_is_interactive_only_when_flagged():
    assert A._is_interactive({"interactive": True}) is True
    assert A._is_interactive({"interactive": False}) is False
    assert A._is_interactive({}) is False  # eval/one-shot по умолчанию тугой


def test_interactive_runs_plan_to_natural_end_no_early_cut():
    """Интерактив: при исчерпанном МЯГКОМ бюджете (150с/8 шагов) НЕ рубим — даём плану доработать."""
    state = {"interactive": True, "steps_executed": 8,  # > MAX_STEPS_PER_RUN(8), но < HARD(40)
             "current_step": 2, "subtasks": [{}, {}, {}, {}]}  # ещё есть подшаги
    with patch.object(A.runbudget, "elapsed", return_value=300.0):  # > 150с мягкого, < 1200 hard
        # eval бы остановился, интерактив — продолжает шаг
        assert A.route_after_step(state) == "step_executor"


def test_interactive_stops_at_hard_ceiling():
    """Интерактив: абсолютный предохранитель (hard) всё же останавливает — анти-патология."""
    state = {"interactive": True, "steps_executed": 41,  # > MAX_STEPS_HARD(40)
             "current_step": 2, "subtasks": [{}, {}, {}, {}]}
    with patch.object(A.runbudget, "elapsed", return_value=100.0):
        assert A.route_after_step(state) == "synthesize"
    # и по времени
    state2 = {"interactive": True, "steps_executed": 5, "current_step": 2, "subtasks": [{}, {}, {}]}
    with patch.object(A.runbudget, "elapsed", return_value=A.MAX_RUN_SECONDS_HARD + 1):
        assert A.route_after_step(state2) == "synthesize"


def test_eval_keeps_tight_budget():
    """Eval/one-shot (interactive=False): тугой потолок 8 шагов рубит как раньше."""
    state = {"steps_executed": 8, "current_step": 2, "subtasks": [{}, {}, {}]}
    with patch.object(A.runbudget, "elapsed", return_value=10.0), \
         patch.object(A.runbudget, "exhausted", return_value=False), \
         patch.object(A.runbudget, "used", return_value=0):
        # steps_executed >= MAX_STEPS_PER_RUN → synthesize (тугой бюджет)
        assert A.route_after_step(state) == "synthesize"
