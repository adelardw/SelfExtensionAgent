"""Фиксы по раунду 3 валидации: анти-фабрикация «из статьи» (директива + учёт чтений),
circuit-breaker поиска, фенс-толерантный парс консенсуса. Offline."""
import importlib.util

import pytest

from src.runtime import run_context as rc


# ── анти-фабрикация: чистая директива + учёт чтений ────────────────────────────

def test_unread_source_directive_gating():
    from src.graph.agent import _unread_source_directive as d

    assert d(True, 0)                                  # doc-интент + ноль чтений → запрет
    assert "НЕ ПРОЧИТАН" in d(True, 0) and "Table" in d(True, 0)
    assert d(True, 2) == ""                            # документ читали → директива не нужна
    assert d(False, 0) == ""                           # обычный вопрос → память легитимна


def test_page_read_stats_scoped_and_cleaned():
    with rc.request_scope("run_r1", "u"):
        rc.note_page_read(True)
        rc.note_page_read(False)
        assert rc.page_read_stats() == (2, 1)
    with rc.request_scope("run_r1", "u"):
        assert rc.page_read_stats() == (0, 0)          # cleanup по выходе


@pytest.fixture
def ws(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "ws_r3", "src/skills/web_search/web_search.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_SEARXNG", "http://stub:8080")
    monkeypatch.setattr(m, "_CLOAK", False)
    monkeypatch.setenv("SEARXNG_DOWN_UNTIL", "0")
    return m


def test_browse_notes_page_reads(ws, monkeypatch):
    monkeypatch.setattr(ws, "_page_text", lambda u: ("Заголовок", "х" * 500))
    with rc.request_scope("run_r2", "u"):
        out = ws.browse.invoke({"url": "https://example.com", "find": ""})
        assert "Заголовок" in out
        assert rc.page_read_stats() == (1, 1)
    monkeypatch.setattr(ws, "_page_text", lambda u: ("", ""))
    with rc.request_scope("run_r3", "u"):
        ws.browse.invoke({"url": "https://example.com", "find": ""})
        assert rc.page_read_stats() == (1, 0)          # пустая страница = неуспех чтения


# ── circuit-breaker поиска ─────────────────────────────────────────────────────

def test_search_circuit_breaker_trips_after_failures(ws, monkeypatch):
    """3 полных провала → дальнейшие вызовы отвечают сентинелом МГНОВЕННО, без сети."""
    monkeypatch.setattr(ws, "_search_searxng", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_search_fallback", lambda *a, **k: [])
    with rc.request_scope("run_cb", "u"):
        for _ in range(3):
            ws.search_web.invoke({"query": "x", "max_results": 3})
        # цепь разомкнута: сетевые бэкенды больше НЕ дёргаются (упадут, если дёрнут)
        def boom(*a, **k):
            raise AssertionError("сеть дёрнули при разомкнутой цепи")
        monkeypatch.setattr(ws, "_search_searxng", boom)
        monkeypatch.setattr(ws, "_search_fallback", boom)
        out = ws.search_web.invoke({"query": "y", "max_results": 3})
        assert "ПОИСК НЕ ДАЛ РЕЗУЛЬТАТОВ" in out and "отключён до конца прогона" in out
        assert rc.search_stats() == (4, 0)


def test_search_attempt_counted_on_entry(ws, monkeypatch):
    """Р.5: попытка считается НА ВХОДЕ — даже если вызов умирает исключением/отменой до
    пост-нотации (wait_for research отменял зависший DDG → цепь не размыкалась)."""
    def boom(*a, **k):
        raise RuntimeError("hang-imitation")

    monkeypatch.setattr(ws, "_search_searxng", boom)
    monkeypatch.setattr(ws, "_search_fallback", boom)
    with rc.request_scope("run_entry", "u"):
        for _ in range(3):
            ws.search_web.invoke({"query": "x", "max_results": 3})
        assert rc.search_stats() == (3, 0)              # все попытки посчитаны
        out = ws.search_web.invoke({"query": "y", "max_results": 3})
        assert "отключён до конца прогона" in out       # цепь разомкнулась


def test_search_circuit_stays_closed_on_success(ws, monkeypatch):
    """Хоть один успех — цепь не размыкается."""
    monkeypatch.setattr(ws, "_search_searxng", lambda *a, **k: [])
    monkeypatch.setattr(ws, "_search_fallback",
                        lambda *a, **k: [{"title": "t", "url": "https://a.ru", "snippet": "s"}])
    with rc.request_scope("run_cb2", "u"):
        for _ in range(4):
            out = ws.search_web.invoke({"query": "x", "max_results": 3})
        assert "a.ru" in out                            # 4-й вызов по-прежнему ищет
        assert rc.search_stats() == (4, 4)


# ── р.4: ЖЁСТКИЙ энфорсмент анти-фабрикации (не промпт) ───────────────────────

def test_doc_source_unread_named_source(monkeypatch):
    """«Прочитал что-то» ≠ «прочитал НАЗВАННЫЙ источник» (дыра р.4)."""
    from src.graph.agent import _doc_source_unread

    q = "приведи числа из статьи arXiv 2310.11511"
    with rc.request_scope("run_ns1", "u"):
        assert _doc_source_unread(q)                    # чтений нет вовсе
    with rc.request_scope("run_ns2", "u"):
        rc.note_page_read(True, "https://random-blog.com/rag-overview")
        assert _doc_source_unread(q)                    # читали НЕ статью → всё ещё unread
    with rc.request_scope("run_ns3", "u"):
        rc.note_page_read(True, "https://ar5iv.labs.arxiv.org/html/2310.11511")
        assert not _doc_source_unread(q)                # named-источник прочитан
    with rc.request_scope("run_ns4", "u"):
        rc.note_page_read(True, "https://any-page.com/x")
        assert not _doc_source_unread("дай точные значения метрик из этой работы")
        # источник НЕ назван явно → достаточно факта содержательного чтения


def test_enforce_replaces_fabricated_attribution():
    from src.graph.agent import _enforce_no_fake_attribution as enf

    q = "числа PopQA из статьи 2310.11511"
    fab = "Согласно Таблице 2 статьи, PopQA составляет 45.0 (7B) и 47.3 (13B)."
    with rc.request_scope("run_enf1", "u"):            # ничего не читали
        out = enf(fab, q, True)
        assert "не могу привести точные значения" in out.lower() and "45.0" not in out
        honest = "Не удалось прочитать статью, поэтому числа привести не могу."
        assert enf(honest, q, True) == honest           # честный отказ проходит нетронутым
        assert enf(fab, q, False) == fab                # не doc-интент → не трогаем
    with rc.request_scope("run_enf2", "u"):
        rc.note_page_read(True, "https://arxiv.org/abs/2310.11511")
        # источник прочитан, но заявленных значений НЕТ в прочитанном (abstract ≠ таблица,
        # финальная дыра р.5) → всё равно замена
        assert "45.0" not in enf(fab, q, True, evidence="abstract текст без чисел")
        # источник прочитан И значения ВИДНЫ в прочитанном → легитимно, не трогаем
        ev = "Table 2: PopQA 45.0 (7B), 47.3 (13B)"
        assert enf(fab, q, True, evidence=ev) == fab


def test_enforce_catches_english_attribution():
    from src.graph.agent import _enforce_no_fake_attribution as enf

    with rc.request_scope("run_enf3", "u"):
        out = enf("According to the paper, Table 3 reports 78.9 accuracy.",
                  "give exact numbers from arXiv 2310.11511", True)
        assert "78.9" not in out


# ── ремонт по расширенной матрице: clarify-замыкание, цены, value-гейт ─────────

def test_clarify_questions_formatting():
    from src.graph.agent import _format_clarify_questions

    out = _format_clarify_questions([
        {"question": "Что сравниваем?", "options": ["CRM", "почтовые сервисы"]},
        {"question": "Для кого презентация?", "options": []},
    ])
    assert "1. Что сравниваем? (варианты: CRM, почтовые сервисы)" in out
    assert "2. Для кого презентация?" in out
    assert "уточни" in out.lower() and "продолжу" in out.lower()


def test_grounded_numbers_catch_thousand_separated_prices():
    """Ценовая фабрикация матрицы: «1 290 ₽» со ссылкой на страницу, где только 2 490 ₽."""
    from src.graph.agent import _attr_values_ungrounded as f

    assert f("Базовый — 1 290 ₽/мес согласно странице тарифов",
             "на странице: тариф 2 490 ₽ в месяц")
    assert not f("тариф 2 490 ₽ согласно странице", "страница: 2 490 ₽ в месяц")
    assert not f("ответ без сумм и метрик", "что угодно")


def test_doc_extraction_seeds_cover_pricing_pages():
    import pytest as _pt

    from src.graph.semantic_signals import _DOC_POS

    assert any("тариф" in s or "стоит" in s for s in _DOC_POS)  # прайс-фразы в сидax
    _ = _pt  # (сид-проверка структурная; embedding-срабатывание — в live-логах)


# ── фенс-толерантный парс консенсус-судьи ──────────────────────────────────────

def test_parse_fenced_model_variants():
    from src.graph.agent import _parse_fenced_model
    from src.llm.structured_outputs import ValidationResult

    body = ('{"is_valid": true, "confidence": 0.8, "feedback": "ок", '
            '"false_refusal": false, "meta_stub": false, "false_completion": false}')
    for text in (f"```json\n{body}\n```", f"```\n{body}\n```", body):
        v = _parse_fenced_model(ValidationResult, text)
        assert v.is_valid and v.confidence == pytest.approx(0.8)
    with pytest.raises(Exception):
        _parse_fenced_model(ValidationResult, "это не json")
