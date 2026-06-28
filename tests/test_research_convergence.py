"""Сходимость research-цикла шага (требование «сходиться должно»): кап вызовов + стоп на «сухих».

Чистые хелперы `_is_research_tool`/`_research_dry`/`_research_exhausted` — БЕЗ API. Гарантируют, что
шаг не кружит по вебу бесконечно: либо исчерпан кап research-вызовов, либо 2 «сухих» раунда подряд.
"""
import src.graph.agent as A


def test_is_research_tool():
    assert A._is_research_tool("deep_research")
    assert A._is_research_tool("run_researcher")
    assert not A._is_research_tool("search_web")
    assert not A._is_research_tool(None)


def test_research_dry_marker():
    assert A._research_dry("[research: 0/3 под-вопросов подтверждено]\n…") is True
    assert A._research_dry("[research: 2/3 под-вопросов подтверждено]\n…") is False
    assert A._research_dry("обычный вывод без маркера") is False
    assert A._research_dry("") is False


def test_research_exhausted_by_cap():
    assert A._research_exhausted(A._RESEARCH_CALL_CAP, 0) is True   # исчерпан кап вызовов
    assert A._research_exhausted(A._RESEARCH_CALL_CAP - 1, 0) is False


def test_research_exhausted_by_dry():
    assert A._research_exhausted(1, 2) is True     # 2 «сухих» подряд → хватит
    assert A._research_exhausted(1, 1) is False    # один сухой — ещё ищем
    assert A._research_exhausted(0, 0) is False     # свежий старт


def test_convergence_guaranteed():
    """Гарантия СХОДИМОСТИ: при любом наборе раундов в какой-то момент СТОП (нет кружения)."""
    calls = dry = 0
    for _ in range(50):                             # имитируем раунды; копим худший случай
        if A._research_exhausted(calls, dry):
            break
        calls += 1
        dry += 1
    else:
        raise AssertionError("research-цикл не остановился — кружение")
