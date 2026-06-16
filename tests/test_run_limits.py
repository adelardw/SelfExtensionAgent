"""Бюджет прогона ПО ТИПУ ЗАДАЧИ (полировка ядра): код/агентные задачи получают ×mult,
простой research — базовый. КЛЮЧЕВОЕ: research НЕ трогается → GAIA не регрессирует by construction."""
import src.agent as a


def test_research_keeps_base_budget():
    base = (a.MAX_RUN_TOKENS, a.MAX_RUN_SECONDS)
    assert a._run_limits({"selected_skills": ["web_search"]}) == base   # research → база
    assert a._run_limits({"selected_skills": ["file_operations"]}) == base
    assert a._run_limits({}) == base                                     # пусто → база
    assert a._run_limits({"selected_skills": []}) == base


def test_agentic_gets_multiplier():
    base = (a.MAX_RUN_TOKENS, a.MAX_RUN_SECONDS)
    m = a.AGENTIC_BUDGET_MULT
    for sk in ("code", "device_control", "browser_control", "app_control"):
        tl, sl = a._run_limits({"selected_skills": [sk]})
        assert tl == int(base[0] * m) and sl == base[1] * m, sk


def test_mixed_selection_agentic_wins():
    base = (a.MAX_RUN_TOKENS, a.MAX_RUN_SECONDS)
    tl, _ = a._run_limits({"selected_skills": ["web_search", "code"]})  # есть агентный → ×mult
    assert tl == int(base[0] * a.AGENTIC_BUDGET_MULT)


def test_step_hard_cut_above_soft_ceiling():
    """ВНУТРИ-шаговый обрыв вооружается ВЫШЕ мягкого потолка (×STEP_HARD_CUT_MULT): граничные
    прогоны (между 1.0× и Nx) рвёт мягкий между-нодовый потолок, а не жёсткий arm → нет потери
    качества последнего шага; arm ловит только интра-степ взрыв. Провабельная нейтральность."""
    st = {"selected_skills": ["web_search"]}        # research → база
    soft_tl, soft_sl = a._run_limits(st)
    hard_tl, hard_sl = a._step_hard_limits(st)
    assert a.STEP_HARD_CUT_MULT >= 1.5              # запас над мягким потолком
    assert hard_tl == int(soft_tl * a.STEP_HARD_CUT_MULT)
    assert hard_sl == soft_sl * a.STEP_HARD_CUT_MULT
    assert hard_tl > soft_tl                        # жёсткий обрыв строго выше мягкого
