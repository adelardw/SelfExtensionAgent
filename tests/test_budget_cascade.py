"""Каскад ярусов + доведение БЕЗ бюджета.
- Финальную сборку делает УМНАЯ модель (deep), легвóрк — дёшево.
- Бюджет (время/токены/8-шагов) УБРАН: план самоограничен → доводим до естественного конца.
- Единственный ранний выход — абсолютный анти-runaway потолок шагов.
"""
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


def test_no_budget_runs_to_natural_completion():
    """Бюджет УБРАН: даже на долгом прогоне (много шагов/времени) НЕ рубим — остались подшаги → шаг."""
    state = {"steps_executed": 25, "current_step": 2, "subtasks": [{}, {}, {}, {}]}  # 25 шагов, <HARD
    assert A.route_after_step(state) == "step_executor"  # никаких 8-шагов/150с-резов


def test_route_synthesize_when_plan_done():
    """План естественно закончился (все подшаги пройдены) → синтез."""
    state = {"steps_executed": 5, "current_step": 4, "subtasks": [{}, {}, {}, {}]}
    assert A.route_after_step(state) == "synthesize"


def test_anti_runaway_step_ceiling():
    """Абсолютный анти-runaway потолок шагов (патология, НЕ бюджет) — единственный ранний выход."""
    state = {"steps_executed": A.MAX_STEPS_HARD, "current_step": 1, "subtasks": [{}, {}, {}]}
    assert A.route_after_step(state) == "synthesize"
