"""База знаний: иерархия папок + гибридный retrieval (граф+BM25). Офлайн (LLM-граф замокан)."""
import tempfile
from pathlib import Path

import pytest

import src.knowledge_base as KB


@pytest.fixture(autouse=True)
def _force_bm25(monkeypatch):
    # Офлайн: форсируем BM25-фолбэк (LightRAG требует сети/ключа). Граф-путь тестируется живьём.
    import src.lightrag_engine as LR
    monkeypatch.setattr(LR, "lightrag_available", lambda: False)


def _doc(text: str, name: str) -> str:
    p = Path(tempfile.mkdtemp()) / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_add_search_with_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(KB, "_KB_ROOT", tmp_path / "kb")
    uid = "u1"
    assert not KB.kb_has_docs(uid)
    KB.create_folder(uid, "work/reports")
    src = _doc("Проект Орион: квартальная выручка составила 4.2 млн евро в Q3.\n\n"
               "Команда выросла до 18 человек. Следующий релиз — в ноябре.", "orion.txt")
    msg = KB.add_document(uid, src, folder="work/reports")
    assert "Добавлено" in msg and KB.kb_has_docs(uid)
    # реальная иерархия папок на диске
    assert (tmp_path / "kb" / "u1" / "work" / "reports" / "orion.txt").exists()
    # retrieval находит релевантный фрагмент
    out = KB.search_kb(uid, "какая выручка проекта Орион")
    assert "4.2 млн" in out and "orion.txt" in out


def test_folder_scoped_search(monkeypatch, tmp_path):
    monkeypatch.setattr(KB, "_KB_ROOT", tmp_path / "kb")
    uid = "u2"
    KB.add_document(uid, _doc("Рецепт борща: свёкла, капуста, картофель.", "borscht.txt"), folder="recipes")
    KB.add_document(uid, _doc("Налоговый отчёт за 2025 год, ставка 13%.", "tax.txt"), folder="finance")
    out = KB.search_kb(uid, "борщ свёкла", folder="recipes")
    assert "борща" in out and "tax" not in out


def test_empty_kb(monkeypatch, tmp_path):
    monkeypatch.setattr(KB, "_KB_ROOT", tmp_path / "kb")
    assert "пуст" in KB.list_kb("nobody").lower() or "ничего" in KB.search_kb("nobody", "x").lower()


def test_tool_builds(monkeypatch, tmp_path):
    import asyncio
    monkeypatch.setattr(KB, "_KB_ROOT", tmp_path / "kb")
    KB.add_document("u3", _doc("Секретный пароль от сейфа: 7788.", "note.txt"))
    t = KB.make_kb_tool("u3")
    assert t.name == "search_knowledge_base"
    assert "7788" in asyncio.run(t.ainvoke({"query": "пароль от сейфа"}))


def test_session_files_temporary(monkeypatch, tmp_path):
    monkeypatch.setattr(KB, "_SESSION_ROOT", tmp_path / "sess")
    sid = "sess-123"
    assert not KB.session_has_files(sid)
    KB.add_session_file(sid, _doc("Договор №42: срок поставки 14 дней, штраф 0.1% в день.", "contract.txt"))
    assert KB.session_has_files(sid)
    out = KB.search_session(sid, "срок поставки по договору")
    assert "14 дней" in out
    # ярус 3 НЕ сохраняется — clear стирает
    KB.clear_session(sid)
    assert not KB.session_has_files(sid)


def test_session_isolated_from_kb(monkeypatch, tmp_path):
    monkeypatch.setattr(KB, "_KB_ROOT", tmp_path / "kb")
    monkeypatch.setattr(KB, "_SESSION_ROOT", tmp_path / "sess")
    KB.add_session_file("s1", _doc("Временная заметка сессии: код 999.", "tmp.txt"))
    # постоянная БЗ юзера s1 при этом пуста (сессия ≠ persist)
    assert not KB.kb_has_docs("s1")


def test_folder_traversal_stays_inside_root(monkeypatch, tmp_path):
    monkeypatch.setattr(KB, "_KB_ROOT", tmp_path / "kb")
    uid = "u4"
    # '..' и абсолютные пути не выводят за корень юзера
    rel = KB.create_folder(uid, "../../evil")
    assert ".." not in rel
    KB.add_document(uid, _doc("данные", "x.txt"), folder="../../../tmp/evil")
    root = (tmp_path / "kb" / "u4").resolve()
    stored = [p for p in root.rglob("x.txt")]
    assert stored and all(p.resolve().is_relative_to(root) for p in stored)
    assert not (tmp_path / "tmp").exists() and not (tmp_path / "evil").exists()


def test_raw_search_structured_none(monkeypatch, tmp_path):
    """Внутренний контракт для recall: None вместо фраз-сентинелей («не нашлось…»)."""
    import asyncio
    monkeypatch.setattr(KB, "_KB_ROOT", tmp_path / "kb")
    monkeypatch.setattr(KB, "_SESSION_ROOT", tmp_path / "sess")
    assert asyncio.run(KB.search_kb_raw("empty-user", "что угодно")) is None
    assert KB.search_session_raw("empty-sess", "что угодно") is None
    KB.add_document("u5", _doc("Ставка по контракту — 7 процентов годовых.", "rate.txt"))
    hit = asyncio.run(KB.search_kb_raw("u5", "ставка по контракту"))
    assert hit and "7 процентов" in hit


def test_autorag_cheap_path_skips_graph(monkeypatch, tmp_path):
    """use_graph=False (авто-впрыск recall на каждый запрос) не дёргает LightRAG вовсе."""
    import asyncio
    import src.lightrag_engine as LR
    monkeypatch.setattr(KB, "_KB_ROOT", tmp_path / "kb")

    async def _boom(*a, **kw):
        raise AssertionError("LightRAG не должен вызываться при use_graph=False")
    monkeypatch.setattr(LR, "query", _boom)
    KB.add_document("u6", _doc("Дедлайн проекта Альфа — 3 марта.", "alpha.txt"))
    hit = asyncio.run(KB.search_kb_raw("u6", "дедлайн проекта Альфа", use_graph=False))
    assert hit and "3 марта" in hit


def test_add_without_graph_skips_lightrag(monkeypatch, tmp_path):
    """use_graph=False (юзер отказался платить): документ в BM25, LR.insert не дёргается."""
    import src.lightrag_engine as LR
    monkeypatch.setattr(KB, "_KB_ROOT", tmp_path / "kb")

    async def _boom(*a, **kw):
        raise AssertionError("LightRAG.insert не должен вызываться при use_graph=False")
    monkeypatch.setattr(LR, "insert", _boom)
    import asyncio
    msg = asyncio.run(KB.add_document_async("u7", _doc("Бюджет отдела — 12 тысяч.", "b.txt"),
                                            use_graph=False))
    assert "граф пропущен" in msg and KB.kb_has_docs("u7")
    assert "12 тысяч" in KB.search_kb("u7", "бюджет отдела")


def test_estimate_index_cost_scales():
    from src.lightrag_engine import estimate_index_cost
    small, big = estimate_index_cost("раз " * 200), estimate_index_cost("раз " * 20000)
    assert small["usd"] > 0 and big["usd"] > small["usd"] and big["chunks"] > small["chunks"]
