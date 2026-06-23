"""Мульти-роль профиль (теги+recency, без отдельной подсистемы), стэши, thrash-маркер."""
import importlib.util
import sys
import time

import pytest


@pytest.fixture()
def store(tmp_path):
    from src.memory.store import MemoryStore

    s = MemoryStore(str(tmp_path / "m.db"))
    yield s
    s.close()


def test_multirole_from_tagged_facts(store):
    store.add_fact("u", "профессия", "фин-аналитик", importance=0.9, tags=["роль"])
    store.add_fact("u", "вторая роль", "разработчик", importance=0.9, tags=["роль", "стек"])
    store.add_fact("u", "город", "Алматы", importance=0.7, tags=["личное"])
    roles = store.get_role_facts("u")
    vals = {f["value"] for f in roles}
    assert vals == {"фин-аналитик", "разработчик"}  # обе роли, город не роль
    block = store.format_profile("u")
    # профиль — ВНУТРЕННЕЕ состояние: содержит роли, но велит НЕ упоминать их в ответе
    assert "фин-аналитик" in block and "разработчик" in block
    assert "ВНУТРЕННЕЕ" in block and "НЕ упоминай" in block


def test_no_profile_no_block(store):
    store.add_fact("u", "город", "Алматы", tags=["личное"])
    assert store.format_profile("u") == ""


def test_role_hygiene_via_recency(store, monkeypatch):
    """Заброшенная роль (старый ts) тонет по recency ниже свежей — гигиена без порогов."""
    store.add_fact("u", "старая роль", "дизайнер", importance=0.9, tags=["роль"])
    # сделаем её «старой»: подменим ts напрямую
    store._conn.execute("UPDATE facts SET ts=ts-? WHERE value='дизайнер'", (40 * 86400,))
    store._conn.commit()
    store.add_fact("u", "новая роль", "разработчик", importance=0.9, tags=["роль"])
    roles = store.get_role_facts("u", k=2)
    assert roles[0]["value"] == "разработчик"  # свежая роль выше старой


def test_conflict_overwrites_same_key(store):
    """Смена ситуации: тот же ключ перезаписывается (upsert) — устаревшее уходит."""
    store.add_fact("u", "город", "Москва", tags=["личное"])
    store.add_fact("u", "город", "Алматы", tags=["личное"])  # переехал
    facts = {f["key"]: f["value"] for f in store.get_facts("u")}
    assert facts["город"] == "Алматы" and len([f for f in store.get_facts("u") if f["key"] == "город"]) == 1


# ── стэш ───────────────────────────────────────────────────────────────

@pytest.fixture()
def stash(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_STASH_DIR", str(tmp_path / "stashes"))
    spec = importlib.util.spec_from_file_location("stash_test", "src/skills/stash/stash.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["stash_test"] = m
    spec.loader.exec_module(m)
    return m


def test_stash_add_view_aggregate(stash):
    stash.stash_add.invoke({"name": "бюджет", "row_json": '{"сумма":1500,"категория":"еда"}'})
    stash.stash_add.invoke({"name": "бюджет", "row_json": '{"сумма":3000,"категория":"аренда"}'})
    stash.stash_add.invoke({"name": "бюджет", "row_json": '{"сумма":800,"категория":"еда"}'})
    assert "3 записей" in stash.stash_view.invoke({"name": "бюджет"})
    agg = stash.stash_aggregate.invoke({"name": "бюджет", "field": "сумма", "op": "sum", "group_by": "категория"})
    assert "аренда: 3000" in agg and "еда: 2300" in agg
    total = stash.stash_aggregate.invoke({"name": "бюджет", "field": "сумма", "op": "sum"})
    assert "5300" in total


def test_stash_invalid_json(stash):
    assert "Невалидный JSON" in stash.stash_add.invoke({"name": "x", "row_json": "не json"})


def test_stash_export_csv(stash):
    # Теперь экспорт ОТДАЁТ ФАЙЛ (артефакт), а не CSV-текст в ответе (живой eval: текст = старый дефект).
    from src.runtime import run_context, artifacts
    stash.stash_add.invoke({"name": "t", "row_json": '{"a":1,"b":2}'})
    with run_context.request_scope("stash-test", "local"):
        msg = stash.stash_export_csv.invoke({"name": "t"})
        assert "файл" in msg.lower() and "доставлен" in msg.lower()  # сообщает о доставке, не дублирует CSV
        arts = run_context.artifacts()
        assert len(arts) == 1
        content = artifacts.resolve_artifact(arts[0]["id"]).read_text()
        assert "a" in content and "b" in content and "1" in content  # данные реально в файле


def test_stash_empty(stash):
    assert "пуст" in stash.stash_view.invoke({"name": "нет_такого"})


def test_stash_write_merges_same_form_keeps_other(stash):
    """Запись синонимами (бюджет/расходы) сливается; данные другой формы — отдельно."""
    stash.stash_add.invoke({"name": "бюджет", "row_json": '{"сумма":4500,"категория":"продукты"}'})
    stash.stash_add.invoke({"name": "расходы", "row_json": '{"сумма":3000,"категория":"аренда"}'})
    stash.stash_add.invoke({"name": "мой бюджет", "row_json": '{"сумма":800,"категория":"кофе"}'})
    stash.stash_add.invoke({"name": "задачи", "row_json": '{"задача":"молоко","срок":"завтра"}'})
    # бюджет собрал все 3 расхода, задачи — отдельный стэш
    total = stash.stash_aggregate.invoke({"name": "бюджет", "field": "сумма", "op": "sum"})
    assert "8300" in total
    assert "молоко" in stash.stash_view.invoke({"name": "задачи"})


def test_stash_fuzzy_name_resolve(stash):
    """Аналитика находит стэш, даже если агент назвал его чуть иначе (eval-баг: 0%)."""
    stash.stash_add.invoke({"name": "бюджет", "row_json": '{"сумма":4500,"категория":"еда"}'})
    # спрашиваем под другим именем → должен найти существующий 'бюджет'
    assert "4500" in stash.stash_aggregate.invoke({"name": "мой_бюджет", "field": "сумма", "op": "sum"})
    assert "еда" in stash.stash_view.invoke({"name": "расходы"})
    # единственный стэш → берётся даже при совсем другом имени
    assert "4500" in stash.stash_aggregate.invoke({"name": "финансы", "field": "сумма"})


# ── thrash-маркер отказа HITL ───────────────────────────────────────────

def test_refusal_marker_constant():
    from src.runtime.hitl import REFUSAL_MARK

    assert REFUSAL_MARK and "ОТКЛОНЕНО" in REFUSAL_MARK
