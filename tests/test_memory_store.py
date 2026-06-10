"""MemoryStore (SQLite): эпизоды/факты/рёбра/цели/recall/prune — без LLM и эмбеддингов."""
import pytest

from src.memory.store import MemoryStore

UID = "test-user"


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "memory.db"))
    yield s
    s.close()


def test_episode_failures_successes(store):
    store.add_episode(UID, "сломалось", "не вышло", outcome="fail", run_id="r1", mode="deliberate")
    store.add_episode(UID, "получилось", "готово", outcome="ok", confidence=0.9, run_id="r2", mode="deliberate")
    store.add_episode(UID, "привет", "привет!", outcome="ok", confidence=0.0, mode="fast")

    fails = store.get_failures()
    assert len(fails) == 1 and fails[0]["query"] == "сломалось"
    # успех = ok И confidence>0 (fast-эпизоды с conf=0 не считаются)
    sucs = store.get_successes()
    assert len(sucs) == 1 and sucs[0]["query"] == "получилось"
    assert store.failure_count() == 1


def test_fact_upsert_and_tags(store):
    fid1 = store.add_fact(UID, "имя", "Жас", tags=["личное"])
    fid2 = store.add_fact(UID, "ИМЯ", "Жас-обновлённый")  # upsert регистронезависимый
    assert fid1 == fid2
    facts = store.get_facts(UID)
    assert len(facts) == 1 and facts[0]["value"] == "Жас-обновлённый"


def test_edges_neighbors(store):
    ep = store.add_episode(UID, "q", "a")
    fact = store.add_fact(UID, "стек", "python", tags=["стек"])
    store.add_edge(UID, "episode", ep, "fact", fact, relation="derived")
    nb = store.neighbors(UID, "episode", ep)
    assert any(r["type"] == "fact" and r["id"] == fact for r in nb)


def test_goal_lifecycle(store):
    store.set_goal(UID, "выучить langgraph", criteria=["пройден туториал"])
    g = store.get_active_goal(UID)
    assert g is not None and "langgraph" in g["aim"]
    assert store.goal_criteria(g) == ["пройден туториал"]
    store.close_active_goal(UID)
    assert store.get_active_goal(UID) is None


def test_recall_budget(store):
    for i in range(10):
        store.add_fact(UID, f"факт{i}", "значение " * 50)
    ctx = store.recall(UID, "значение", budget=500)
    assert isinstance(ctx, str)
    # бюджет соблюдается с разумным допуском на заголовки секций
    assert len(ctx) <= 900


def test_mode_stats(store):
    store.add_episode(UID, "q1", "a", mode="fast")
    store.add_episode(UID, "q2", "a", mode="fast")
    store.add_episode(UID, "q3", "a", mode="deliberate", confidence=0.9)
    store.add_episode(UID, "q4", "a", mode="clarify")
    ms = store.mode_stats(UID)
    assert ms["total"] == 4
    assert ms["modes"] == {"fast": 2, "deliberate": 1, "clarify": 1}
    assert ms["fast_share"] == 0.5
    assert ms["cheap_share"] == 0.75


def test_prune(store):
    for i in range(20):
        store.add_episode(UID, f"q{i}", "a")
    removed = store.prune(max_episodes=5, max_facts=300, max_reflections=200)
    assert isinstance(removed, dict)
    assert store.episode_count(UID) <= 5
