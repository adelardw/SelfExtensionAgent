"""Трек B: память-как-tool — search_memory + recall_history (drill-back). Офлайн."""
import pytest

from src.memory_tools import make_memory_tools


@pytest.fixture
def store(tmp_path):
    from src.memory.store import MemoryStore
    return MemoryStore(str(tmp_path / "m.db"))


def test_tools_built():
    from src.memory.store import MemoryStore
    import tempfile
    s = MemoryStore(tempfile.mktemp(suffix=".db"))
    tools = make_memory_tools(s, "u1")
    names = {t.name for t in tools}
    # 3 яруса как тулы: глобальная (search_memory) + drill-back (recall_history) + временная
    assert names == {"search_memory", "recall_history", "note_to_self", "read_my_notes"}


def test_temporary_memory_scratchpad():
    from src.memory.store import MemoryStore
    from src.memory_tools import clear_scratch
    import tempfile
    s = MemoryStore(tempfile.mktemp(suffix=".db"))
    tools = {t.name: t for t in make_memory_tools(s, "u1")}
    tools["note_to_self"].invoke({"note": "ключевой факт: бюджет 5000"})
    out = tools["read_my_notes"].invoke({"query": ""})
    assert "бюджет 5000" in out
    clear_scratch("u1")  # временный ярус не переживает прогон
    assert "бюджет 5000" not in tools["read_my_notes"].invoke({"query": ""})


def test_search_memory_returns_facts(store):
    store.add_fact(user_id="u1", key="любимый язык", value="Python", importance=0.8)
    tools = {t.name: t for t in make_memory_tools(store, "u1")}
    out = tools["search_memory"].invoke({"query": "какой язык предпочитает"})
    assert "Python" in out


def test_recall_history_restores_full_episode(store):
    long_answer = "Подробный ответ про настройку окружения: " + "детали " * 50
    store.add_episode(user_id="u1", query="как настроить окружение проекта",
                      answer=long_answer, confidence=0.9, outcome="ok", run_id="r1")
    tools = {t.name: t for t in make_memory_tools(store, "u1")}
    out = tools["recall_history"].invoke({"query": "настройка окружения"})
    # drill-back возвращает ПОЛНЫЙ ответ, не обрезок-индекс
    assert "Подробный ответ про настройку" in out and out.count("детали") > 40


def test_isolation_between_users(store):
    store.add_fact(user_id="alice", key="секрет", value="яблоко", importance=0.9)
    tools_bob = {t.name: t for t in make_memory_tools(store, "bob")}
    assert "яблоко" not in tools_bob["search_memory"].invoke({"query": "секрет"})
