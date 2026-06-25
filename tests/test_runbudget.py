"""Токен-бюджет прогона + маршрутные предохранители от runaway."""
from src.runtime import runbudget


def test_budget_accumulates_and_resets():
    runbudget.reset()
    assert runbudget.used() == 0
    runbudget.add(1000)
    runbudget.add(500)
    assert runbudget.used() == 1500
    assert runbudget.over(1000) and not runbudget.over(2000)
    runbudget.reset()
    assert runbudget.used() == 0 and not runbudget.over(1)


def test_budget_ignores_negative():
    runbudget.reset()
    runbudget.add(-50)
    assert runbudget.used() == 0


def test_exhausted_by_tokens_or_time():
    runbudget.reset()
    assert not runbudget.exhausted(100, 9999)
    runbudget.add(200)
    assert runbudget.exhausted(100, 9999)          # по токенам
    runbudget.reset()
    assert runbudget.exhausted(10**9, 0)            # по времени (лимит 0с)
    assert runbudget.elapsed() >= 0


def test_callback_counts_token_usage():
    runbudget.reset()
    cb = runbudget.callback()

    class _Resp:
        llm_output = {"token_usage": {"prompt_tokens": 800, "completion_tokens": 200}}

    cb.on_llm_end(_Resp())
    assert runbudget.used() == 1000


def test_route_after_step_ignores_token_budget(monkeypatch):
    """Бюджет УБРАН: «за токен-бюджетом» прогон НЕ рубится — есть подшаги → step_executor."""
    import src.graph.agent as A

    runbudget.reset()
    runbudget.add(10**9)  # «за бюджетом» — на маршрут больше не влияет
    state = {"steps_executed": 1, "current_step": 0, "subtasks": [{"goal": "a"}, {"goal": "b"}]}
    assert A.route_after_step(state) == "step_executor"
    runbudget.reset()


def test_route_after_step_continues_within_budget(monkeypatch):
    import src.graph.agent as A

    monkeypatch.setattr(A, "MAX_RUN_TOKENS", 120000)
    monkeypatch.setattr(A, "MAX_STEPS_PER_RUN", 8)
    runbudget.reset()
    runbudget.add(1000)
    state = {"steps_executed": 1, "current_step": 0, "subtasks": [{"goal": "a"}, {"goal": "b"}]}
    assert A.route_after_step(state) == "step_executor"
    runbudget.reset()


def test_no_step_budget_only_anti_runaway_ceiling(monkeypatch):
    """8-шаговый бюджет убран: много шагов → идём дальше; стоп только на абсолютном MAX_STEPS_HARD."""
    import src.graph.agent as A

    runbudget.reset()
    s1 = {"steps_executed": 10, "current_step": 0, "subtasks": [{"goal": "a"}, {"goal": "b"}]}
    assert A.route_after_step(s1) == "step_executor"          # 10 шагов — не рубим
    s2 = {"steps_executed": A.MAX_STEPS_HARD, "current_step": 0, "subtasks": [{"goal": "a"}]}
    assert A.route_after_step(s2) == "synthesize"             # абсолютный анти-runaway потолок
    runbudget.reset()
