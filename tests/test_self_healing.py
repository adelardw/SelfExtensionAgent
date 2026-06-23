"""Контур самопочинки навыков: per-skill здоровье + детект degraded + repair-гейты."""
import asyncio
import os
import tempfile

import pytest


@pytest.fixture
def in_tmp(monkeypatch):
    old = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    yield
    os.chdir(old)


def test_health_degrades_after_same_class_streak(in_tmp):
    from src.tools import skill_health as H
    H.reset()
    H.record("api_skill", ok=False, err="HTTPError 503")
    H.record("api_skill", ok=False, err="HTTPError 502")
    assert H.health("api_skill")["status"] == "ok"      # 2 < порог
    H.record("api_skill", ok=False, err="HTTP 500")
    assert H.health("api_skill")["status"] == "degraded"  # 3 подряд одного класса
    assert "api_skill" in H.degraded()


def test_health_recovers_on_success(in_tmp):
    from src.tools import skill_health as H
    H.reset()
    for _ in range(3):
        H.record("s", ok=False, err="timeout")
    assert H.health("s")["status"] == "degraded"
    H.record("s", ok=True)                       # внешний сервис ожил
    assert H.health("s")["status"] == "ok"
    assert H.degraded() == []


def test_different_error_classes_dont_accumulate(in_tmp):
    from src.tools import skill_health as H
    H.reset()
    H.record("s", ok=False, err="timeout")
    H.record("s", ok=False, err="ImportError: no module foo")
    H.record("s", ok=False, err="connection refused")
    # три РАЗНЫХ класса → не серия → не degraded (это разовые сбои, не системная поломка)
    assert H.health("s")["status"] == "ok"


def test_records_last_fail_args_for_regression(in_tmp):
    from src.tools import skill_health as H
    H.reset()
    H.record("s", ok=False, err="http 500", args={"q": "x"})
    assert H.health("s")["last_fail_args"] == {"q": "x"}


def test_repair_respects_cap(in_tmp):
    from src.tools import skill_health as H, skill_repair as R
    H.reset()
    H.record("t", ok=False, err="http 500")
    for _ in range(R.MAX_REPAIRS):
        H.mark_repaired("t", success=False)
    out = asyncio.run(R.repair_tool("t"))
    assert out["ok"] is False and "cap" in out["reason"]


def test_repair_no_owner(in_tmp):
    from src.tools import skill_health as H, skill_repair as R
    H.reset()
    H.record("definitely_not_a_real_skill_tool", ok=False, err="x")
    out = asyncio.run(R.repair_tool("definitely_not_a_real_skill_tool"))
    assert out["ok"] is False and "не найден" in out["reason"]


def test_strip_fence():
    from src.tools.skill_repair import _strip_fence
    assert _strip_fence("```python\ncode\n```") == "code"
    assert _strip_fence("plain") == "plain"
