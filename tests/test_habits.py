"""Контур B (само-расширение из повторов): k похожих успешных дорогих прогонов → привычка →
факт-директива create_skill → закрытие после создания навыка. Офлайн, без LLM."""
import pytest

from src import habits
from src.memory.store import MemoryStore

UID = "test-user"
Q = "сделай еженедельный отчёт по продажам из excel в pdf"


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "memory.db"))
    yield s
    s.close()


def _ok_run(store, query, mode="deliberate"):
    store.add_episode(UID, query, "готово", outcome="ok", confidence=0.9, mode=mode)


def test_similar_successes_counts_cluster(store):
    _ok_run(store, Q)
    _ok_run(store, "собери еженедельный отчёт по продажам из excel")
    _ok_run(store, "какая погода в Алматы")  # не кластер
    _ok_run(store, "сделай отчёт по продажам за неделю из excel в pdf")
    assert len(store.similar_successes(UID, Q)) == 3


def test_flag_only_after_k_repeats(store):
    _ok_run(store, Q)
    assert habits.maybe_flag(store, UID, Q, k=3) is None  # 1 раз — не привычка
    _ok_run(store, "собери еженедельный отчёт по продажам из excel в pdf")
    assert habits.maybe_flag(store, UID, Q, k=3) is None  # 2 раза — ещё нет
    _ok_run(store, "сделай отчёт по продажам за эту неделю из excel в pdf")
    key = habits.maybe_flag(store, UID, Q, k=3)
    assert key and key.startswith("привычка: ")
    facts = {f["key"]: f["value"] for f in store.get_facts(UID)}
    assert "create_skill" in facts[key]
    # повторный прогон того же кластера НЕ плодит вторую директиву
    _ok_run(store, Q)
    assert habits.maybe_flag(store, UID, Q, k=3) is None
    assert sum(1 for f in store.get_facts(UID) if f["key"].startswith("привычка")) == 1


def test_fast_and_failed_runs_dont_count(store):
    for _ in range(5):
        store.add_episode(UID, Q, "ответ", outcome="ok", confidence=0.9, mode="fast")
    store.add_episode(UID, Q, "не вышло", outcome="fail", mode="deliberate")
    assert habits.maybe_flag(store, UID, Q, k=3) is None  # дешёвые/провальные ≠ привычка


def test_resolve_closes_habit(store):
    for q in (Q, "собери еженедельный отчёт по продажам из excel в pdf",
              "сделай отчёт по продажам за неделю из excel в pdf"):
        _ok_run(store, q)
    key = habits.maybe_flag(store, UID, Q, k=3)
    assert key
    assert habits.resolve(store, UID, Q, "sales_report_generator") is True
    facts = {f["key"]: f["value"] for f in store.get_facts(UID)}
    assert facts[key].startswith("✅") and "sales_report_generator" in facts[key]
    # закрытая привычка: не флагуется заново и не закрывается повторно
    assert habits.maybe_flag(store, UID, Q, k=3) is None
    assert habits.resolve(store, UID, Q, "another_skill") is False
