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
