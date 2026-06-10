"""Векторизация под пользователя: персональные few-shots + acceptance-маркер feedback."""
from pathlib import Path

import pytest

from src.improve import prompt_store as ps
from src.memory import feedback


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PARAMS_FILE", tmp_path / "params.json")
    monkeypatch.setattr(ps, "USER_FEWSHOTS_FILE", tmp_path / "users.json")


def test_per_user_isolation_and_merge():
    ps.add_user_fewshot("userA", "step_execution", "как считать налог", "доход*0.1", 0.9)
    ps.add_fewshot("step_execution", "глобальный вопрос", "глоб ответ", 0.8)

    a = ps.format_fewshots("step_execution", k=3, user_id="userA")
    assert "налог" in a and "глобальный" in a          # личный + глобальный добивает

    b = ps.format_fewshots("step_execution", k=3, user_id="userB")
    assert "налог" not in b and "глобальный" in b       # чужого личного не видит


def test_personal_takes_priority():
    # k=1: должен показаться ЛИЧНЫЙ, не глобальный
    ps.add_fewshot("step_execution", "g", "global", 0.99)
    ps.add_user_fewshot("u", "step_execution", "personal q", "personal a", 0.5)
    out = ps.format_fewshots("step_execution", k=1, user_id="u")
    assert "personal" in out and "global" not in out


def test_user_fewshot_cap_and_dedup():
    for i in range(ps.MAX_FEWSHOTS + 3):
        ps.add_user_fewshot("u", "step_execution", f"вопрос {i} хвост", f"ответ {i}", i / 10)
    assert len(ps.get_user_fewshots("u", "step_execution", k=99)) <= ps.MAX_FEWSHOTS
    # дедуп по началу запроса
    ps.add_user_fewshot("u", "role2", "один и тот же", "a", 0.3)
    ps.add_user_fewshot("u", "role2", "один и тот же", "b", 0.9)
    shots = ps.get_user_fewshots("u", "role2", k=9)
    assert len(shots) == 1 and shots[0]["answer"] == "b"


def test_lru_user_eviction(monkeypatch):
    monkeypatch.setattr(ps, "MAX_USERS", 3)
    for i in range(5):
        ps.add_user_fewshot(f"user{i}", "step_execution", "q", "a", 0.5)
    data = ps._load_users()
    assert len(data) <= 3                  # старые вытеснены
    assert "user4" in data                 # свежий остался


def test_empty_user_falls_back_to_global():
    ps.add_fewshot("step_execution", "g q", "g a", 0.8)
    # user_id="" → должен вести себя как глобальный харвест
    ps.add_user_fewshot("", "step_execution", "x q", "x a", 0.9)
    assert ps.get_user_fewshots("", "step_execution") == []  # пустой user → ничего личного


def test_feedback_negative_marker():
    neg = feedback._NEG_MARK + " прошлый ответ был слабым"
    pos = "похожая задача уже успешно решалась"
    assert feedback.is_negative(neg) and not feedback.is_negative(pos)
    assert not feedback.strip_marker(neg).startswith(feedback._NEG_MARK)
    assert feedback.strip_marker(pos) == pos
