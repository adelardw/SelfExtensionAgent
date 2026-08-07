"""Фиксы по мульти-агентной валидации: честный отказ поиска (пустота ≠ «темы нет»),
статистика поисковых исходов в run_context, гейт актуальности в синтезе. Offline."""
import importlib.util

import pytest


@pytest.fixture
def ws(monkeypatch):
    """Модуль навыка web_search, загруженный как в проде (spec), с заглушенными бэкендами."""
    spec = importlib.util.spec_from_file_location(
        "ws_test", "src/skills/web_search/web_search.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_SEARXNG", "http://stub:8080")
    monkeypatch.setattr(m, "_CLOAK", False)
    monkeypatch.setenv("SEARXNG_DOWN_UNTIL", "0")
    return m


def test_all_backends_empty_returns_infra_failure(ws, monkeypatch):
    """Пустота ото ВСЕХ бэкендов → сентинел отказа инфраструктуры + запрет выдумки,
    а не размытое «ничего не найдено» (по нему синтез скатывался в устаревшую память)."""
    from src.runtime import run_context as rc

    monkeypatch.setattr(ws, "_search_searxng", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_search_fallback", lambda *a, **k: [])
    with rc.request_scope("run_ws_fail", "u"):
        out = ws.search_web.invoke({"query": "ставка цб", "max_results": 3})
        att, okc = rc.search_stats()
    assert "ПОИСК НЕ ДАЛ РЕЗУЛЬТАТОВ" in out
    assert "не выдавай сведения из памяти за текущие" in out
    assert "апстрим-движки пусты" in out          # различаем «жив, но пуст» от «лежит»
    assert (att, okc) == (1, 0)


def test_empty_searxng_falls_back_to_ddg(ws, monkeypatch):
    """SearXNG жив, но пуст (капча апстримов) → результат приходит из DDG-фолбэка."""
    from src.runtime import run_context as rc

    monkeypatch.setattr(ws, "_search_searxng", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_search_fallback", lambda *a, **k: [
        {"title": "ЦБ снизил ставку", "url": "https://cbr.ru/press/pr/x", "snippet": "14%"}])
    with rc.request_scope("run_ws_ddg", "u"):
        out = ws.search_web.invoke({"query": "ставка цб", "max_results": 3})
        att, okc = rc.search_stats()
    assert "cbr.ru" in out and "http-fallback" in out
    assert (att, okc) == (1, 1)


def test_search_fail_sentinel_parity_with_agent(ws, monkeypatch):
    """Сентинел в web_search и константа агента не должны разъехаться (иначе честный
    шаблон «поиск недоступен» в act перестанет срабатывать)."""
    from src.graph.agent import _SEARCH_FAIL_MARK

    monkeypatch.setattr(ws, "_search_searxng", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_search_fallback", lambda *a, **k: [])
    out = ws.search_web.invoke({"query": "x", "max_results": 3})
    assert _SEARCH_FAIL_MARK in out


def test_search_stats_scoped_and_cleaned():
    from src.runtime import run_context as rc

    with rc.request_scope("run_a", "u"):
        rc.note_search_result(False)
        rc.note_search_result(True)
        assert rc.search_stats() == (2, 1)
    with rc.request_scope("run_b", "u"):
        assert rc.search_stats() == (0, 0)         # чужой прогон не видит
    with rc.request_scope("run_a", "u"):
        assert rc.search_stats() == (0, 0)         # cleanup по выходе из scope


def test_dedup_by_domain():
    """Чтение топа — по одной странице с РАЗНЫХ доменов (живой баг: топ-3 = три клона cbr.ru,
    значение ставки прозой было на источниках ниже и в находки не попадало)."""
    from src.graph.agent import _dedup_by_domain

    urls = ["https://cbr.ru/hd_base/KeyRate/", "https://www.cbr.ru/", "https://cbr.ru/press/",
            "https://ria.ru/rate/", "https://banki.ru/x", "https://gogov.ru/y"]
    assert _dedup_by_domain(urls, 3) == [
        "https://cbr.ru/hd_base/KeyRate/", "https://ria.ru/rate/", "https://banki.ru/x"]
    assert _dedup_by_domain([], 3) == []
    assert _dedup_by_domain(["https://a.ru/1", "https://a.ru/2"], 3) == ["https://a.ru/1"]


def test_stale_directive_gating():
    from src.graph.agent import _stale_data_directive as d

    assert d(3, 0, True)                            # время-чувствительный + все провалились → гейт
    assert "0 успешных" in d(3, 0, True)
    assert "могли устареть" in d(3, 0, True)
    assert d(3, 1, True) == ""                      # хоть один успех → без гейта
    assert d(0, 0, True) == ""                      # поиск не дёргали → это не «провал поиска»
    assert d(3, 0, False) == ""                     # вне времени → память легитимна
