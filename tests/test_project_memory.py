"""Проектный ярус памяти (MEMORY.md, #2): add/recall/index + КЛЮЧЕВОЕ — пустая папка
даёт пустой block() (аддитивно, ноль изменения поведения по умолчанию). Offline, без LLM."""
from pathlib import Path

import pytest

import src.memory.project_memory as pm


@pytest.fixture
def tmp_mem(tmp_path, monkeypatch):
    root = tmp_path / "project_memory"
    monkeypatch.setattr(pm, "_ROOT", root)
    monkeypatch.setattr(pm, "_INDEX", root / "MEMORY.md")
    return root


def test_empty_dir_is_noop(tmp_mem):
    # Нет папки → ничего не подмешивается (это и держит дефолт == бейзлайн).
    assert pm.block("любой запрос") == ""
    assert pm.recall("любой запрос") == ""
    assert pm.index_text() == ""


def test_add_writes_typed_note_and_index(tmp_mem):
    p = pm.add("Профиль юзера", "фин-аналитик, любит ссылки", "Работает в финтехе.", "user")
    assert p.exists() and p.name == "профиль-юзера.md"  # кириллица в slug сохранена
    text = p.read_text(encoding="utf-8")
    assert "type: user" in text and "name: Профиль юзера" in text and "Работает в финтехе." in text
    idx = pm.index_text()
    assert "Профиль юзера" in idx and "(профиль-юзера.md)" in idx  # указатель на slug


def test_concurrent_adds_no_lost_index_lines(tmp_mem):
    """CON-3: параллельные add() не теряют строк-указателей MEMORY.md (RMW под локом + atomic)."""
    import threading

    def w(i: int) -> None:
        pm.add(f"note{i}", f"desc{i}", f"body{i}", "project")

    ths = [threading.Thread(target=w, args=(i,)) for i in range(20)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    idx = pm.index_text()
    lines = [ln for ln in idx.splitlines() if ln.startswith("- [")]
    assert len(lines) == 20  # ни одна строка не затёрта параллельной записью
    assert all(f"(note{i}.md)" in idx for i in range(20))


def test_recall_finds_relevant(tmp_mem):
    pm.add("Бюджет проекта", "тугой бюджет лучше большого", "Гонять GAIA на дефолтном бюджете.", "feedback")
    pm.add("Стек", "питон-агент на langgraph", "Провайдер openrouter.", "project")
    rel = pm.recall("какой бюджет использовать", k=2)
    assert "Бюджет проекта" in rel or "бюджет" in rel.lower()


def test_block_combines_index_and_relevant(tmp_mem):
    pm.add("Цель", "north-star проекта", "Способность=цель, бюджет=констрейнт.", "project")
    b = pm.block("цель проекта")
    assert "[Проектная память — индекс]" in b and "[Релевантные проектные заметки]" in b


def test_bad_type_falls_back_to_project(tmp_mem):
    p = pm.add("X", "desc", "body", "не_тип")
    assert "type: project" in p.read_text(encoding="utf-8")


def test_no_duplicate_index_line(tmp_mem):
    pm.add("Одно", "desc1", "body", "project")
    pm.add("Одно", "desc2", "body2", "project")  # тот же slug
    assert pm.index_text().count("(одно.md)") <= 1  # не дублируем указатель
