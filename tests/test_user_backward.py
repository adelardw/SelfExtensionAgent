"""Per-user backward (gap 1): уроки из неудач юзера → персональные few-shots, изоляция."""
import pytest
from langchain_core.runnables import RunnableLambda

from src.llm.structured_outputs import UserLesson, UserLessons


@pytest.fixture
def store(tmp_path):
    from src.memory.store import MemoryStore
    return MemoryStore(str(tmp_path / "m.db"))


def _seed_failures(store, user_id, n=3):
    for i in range(n):
        store.add_episode(user_id=user_id, query=f"посчитай бюджет за месяц {i}",
                          answer="посчитал в уме неверно", confidence=0.2,
                          outcome="low_conf", run_id=f"{user_id}_r{i}", mode="reason")


def test_per_user_backward_writes_personal_fewshots(store, tmp_path, monkeypatch):
    # изолируем стор few-shots во временный файл
    monkeypatch.setattr("src.improve.prompt_store.USER_FEWSHOTS_FILE", tmp_path / "uf.json")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test")  # пройти guard ключа

    # подменяем LLM: возвращает фиксированный урок, без живого вызова
    lessons = UserLessons(lessons=[UserLesson(
        node="reason", trigger="посчитать бюджет/расходы",
        lesson="не считай в уме — используй навык stash для записи и агрегации")])

    class FakeLLM:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _x: lessons)

    monkeypatch.setattr("src.llm.llm.chat", lambda *a, **k: FakeLLM())
    monkeypatch.setattr("src.llm.llm.provider", lambda: "openrouter")

    _seed_failures(store, "alice", 3)

    from src.improve.graph_learn import graph_backward_user
    res = graph_backward_user(store, "alice", min_batch=3)

    assert res["status"] == "done"
    assert res["batch_size"] == 3
    assert any(s["node"] == "reason" for s in res["lessons_stored"])

    # урок действительно лёг в ПЕРСОНАЛЬНЫЙ стор alice под роль reason
    from src.improve.prompt_store import get_user_fewshots
    shots = get_user_fewshots("alice", "reason", k=5)
    assert shots and "stash" in shots[0]["answer"]
    # изоляция: у другого юзера ничего нет
    assert get_user_fewshots("bob", "reason", k=5) == []


def test_per_user_backward_skips_when_too_few(store, tmp_path, monkeypatch):
    monkeypatch.setattr("src.improve.prompt_store.USER_FEWSHOTS_FILE", tmp_path / "uf.json")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test")
    monkeypatch.setattr("src.llm.llm.provider", lambda: "openrouter")
    _seed_failures(store, "carol", 1)
    from src.improve.graph_learn import graph_backward_user
    res = graph_backward_user(store, "carol", min_batch=3)
    assert res["status"] == "skipped"
