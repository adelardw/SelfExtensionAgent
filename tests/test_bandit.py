"""Контур C: бандит-прайор выбора режима (Beta/Thompson по похожим эпизодам, учитывает
неудачи — которых нет в few-shots). Офлайн, без LLM."""
import random

import pytest

from src import bandit
from src.memory.store import MemoryStore

UID = "test-user"
Q = "проанализируй квартальный отчёт по продажам и сведи таблицу"


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "memory.db"))
    yield s
    s.close()


def test_empty_history_no_prior(store):
    assert bandit.mode_prior(store, UID, Q) == ""


def test_sparse_evidence_no_prior(store):
    store.add_episode(UID, Q, "ок", outcome="ok", mode="fast")
    store.add_episode(UID, "анализ отчёта по продажам за квартал", "ок", outcome="ok", mode="fast")
    assert bandit.mode_prior(store, UID, Q) == ""  # < MIN_EVIDENCE — шум, молчим


def test_negative_evidence_visible_and_converges(store):
    # fast на этом кластере у юзера регулярно проваливается, deliberate работает
    for i in range(4):
        store.add_episode(UID, f"проанализируй отчёт по продажам и сведи таблицу #{i}",
                          "слабо", outcome="fail", mode="fast")
    for i in range(4):
        store.add_episode(UID, f"проанализируй квартальный отчёт по продажам #{i}",
                          "готово", outcome="ok", confidence=0.9, mode="deliberate")
    prior = bandit.mode_prior(store, UID, Q, rng=random.Random(7))
    assert "fast — 0 успех(ов)/4" in prior          # негативное свидетельство видно
    assert "deliberate — 4 успех(ов)/0" in prior
    assert "предлагает «deliberate»" in prior        # Thompson сходится к рабочему режиму
    assert "ПРАЙОР, не приказ" in prior


def test_unrelated_episodes_dont_leak(store):
    for i in range(6):
        store.add_episode(UID, f"какая погода в Алматы сегодня {i}", "солнечно",
                          outcome="ok", mode="fast")
    assert bandit.mode_prior(store, UID, Q) == ""  # чужой кластер — не свидетельство
