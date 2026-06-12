"""Журнал взаимодействий (стадия «сигнал» контура): HITL/clarify-события переживают прогон
и без LLM конвертируются в персонализацию (факты профиля). Офлайн."""
import asyncio
import json

import pytest

from src import hitl, interaction
from src.memory.store import MemoryStore

UID = "test-user"


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "memory.db"))
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _fresh_ledger():
    interaction.reset_ledger()
    yield
    hitl.set_confirmer(None)


def test_hitl_decisions_recorded():
    hitl.set_confirmer(lambda d: False)
    assert asyncio.run(hitl.confirm("device_control.open_app(name=Telegram)")) is False
    hitl.set_confirmer(lambda d: True)
    assert asyncio.run(hitl.confirm("file_operations.write(path=a.txt)")) is True
    evs = interaction.events()
    assert [e["approved"] for e in evs] == [False, True]
    assert evs[0]["type"] == "hitl" and "open_app" in evs[0]["action"]


def test_headless_deny_not_recorded():
    """deny-by-default без канала — не выбор юзера, на нём не учимся."""
    hitl.set_confirmer(None)
    assert asyncio.run(hitl.confirm("device_control.notify(msg=hi)")) is False
    assert interaction.events() == []


def test_harvest_refusal_to_fact(store):
    interaction.record_hitl("device_control.open_app(name=Telegram)", approved=False)
    interaction.record_hitl("file_operations.read(path=a.txt)", approved=True)  # approve ≠ факт
    n = interaction.harvest(store, UID)
    assert n == 1
    facts = {f["key"]: f["value"] for f in store.get_facts(UID)}
    assert "hitl-отказ: device_control.open_app" in facts
    assert "без его явной просьбы" in facts["hitl-отказ: device_control.open_app"]
    # повторный отказ того же тула → upsert, не дубль
    interaction.reset_ledger()
    interaction.record_hitl("device_control.open_app(name=WhatsApp)", approved=False)
    interaction.harvest(store, UID)
    assert sum(1 for f in store.get_facts(UID) if f["key"].startswith("hitl-отказ")) == 1


def test_harvest_clarify_answers_accumulate_onboarding(store):
    items = [
        {"question": "В каком формате отчёт?", "answer": "только PDF", "status": "answered"},
        {"question": "За какой период?", "answer": "Q3", "status": "assumed"},  # допущение ≠ факт
    ]
    n = interaction.harvest(store, UID, clarify_items=items)
    assert n == 1
    facts = {f["key"]: f["value"] for f in store.get_facts(UID)}
    assert facts.get("уточнение: В каком формате отчёт?") == "только PDF"
    assert not any("период" in k for k in facts)


def test_episode_stores_interactions(store):
    evs = [{"type": "hitl", "action": "x.y(z)", "approved": False, "ts": 1.0}]
    ep_id = store.add_episode(UID, "сделай X", "ответ", interactions=evs)
    row = store._conn.execute("SELECT interactions FROM episodes WHERE id=?", (ep_id,)).fetchone()
    assert json.loads(row["interactions"]) == evs


def test_ledger_isolated_between_runs():
    interaction.record_hitl("a.b(c)", approved=False)
    interaction.reset_ledger()  # recall нового прогона
    assert interaction.events() == []
