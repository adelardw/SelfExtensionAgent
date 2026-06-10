"""Реестр уточнений: канал/допущения/ledger/format + маршрутизация clarify_gate."""
import asyncio
import os

import pytest

from src import clarify


@pytest.fixture(autouse=True)
def fresh_ledger():
    clarify.reset_ledger()
    clarify.set_clarifier(None)
    yield
    clarify.set_clarifier(None)


def test_assume_by_default_no_clarifier():
    items = [
        {"question": "Какой формат?", "options": ["pdf", "docx"], "assume": "pdf"},
        {"question": "Сколько страниц?", "options": [], "assume": "1 страница"},
    ]
    resolved = asyncio.run(clarify.ask(items))
    assert resolved[0]["answer"] == "pdf" and resolved[0]["status"] == "assumed"
    assert resolved[1]["answer"] == "1 страница" and resolved[1]["status"] == "assumed"
    assert clarify.has_assumptions() is True


def test_answers_from_clarifier():
    clarify.set_clarifier(lambda items: ["docx", "5"])
    items = [
        {"question": "Формат?", "options": ["pdf", "docx"], "assume": "pdf"},
        {"question": "Страниц?", "options": [], "assume": "1"},
    ]
    resolved = asyncio.run(clarify.ask(items))
    assert [r["answer"] for r in resolved] == ["docx", "5"]
    assert all(r["status"] == "answered" for r in resolved)
    assert clarify.has_assumptions() is False


def test_async_clarifier_and_partial_answers():
    async def chan(items):
        return ["", "вторая"]  # первый не ответили → допущение, второй ответили

    clarify.set_clarifier(chan)
    items = [
        {"question": "A?", "options": [], "assume": "deflt"},
        {"question": "B?", "options": [], "assume": "x"},
    ]
    resolved = asyncio.run(clarify.ask(items))
    assert resolved[0]["answer"] == "deflt" and resolved[0]["status"] == "assumed"
    assert resolved[1]["answer"] == "вторая" and resolved[1]["status"] == "answered"


def test_format_ledger_marks_assumptions():
    # отвечает только на первый вопрос; на остальные — допущение
    clarify.set_clarifier(lambda items: ["да"] if items[0]["question"] == "Точно?" else [""])
    asyncio.run(clarify.ask([{"question": "Точно?", "options": [], "assume": "да"}]))
    asyncio.run(clarify.ask([{"question": "Цвет?", "options": [], "assume": "синий"}]))  # no answer→assumed
    txt = clarify.format_ledger()
    assert "Точно?" in txt and "Цвет?" in txt
    assert "ДОПУЩЕНИЕ" in txt  # второй помечен


def test_reset_isolates_runs():
    asyncio.run(clarify.ask([{"question": "Q", "options": [], "assume": "a"}]))
    assert clarify.ledger()
    clarify.reset_ledger()
    assert clarify.ledger() == []
    assert clarify.format_ledger() == "Уточнений по задаче пока нет."


def test_ask_user_tool_records_to_ledger():
    clarify.set_clarifier(lambda items: ["Алматы"])
    tool = clarify.make_ask_user_tool()
    res = asyncio.run(tool.ainvoke({"question": "Какой город?", "options": "Алматы|Астана"}))
    assert "Алматы" in res
    assert any("город" in it["question"].lower() for it in clarify.ledger())


@pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="нужен API-ключ (llm на импорте agent)",
)
def test_routing_clarify_gate():
    from src.agent import route_after_goal

    assert route_after_goal({"mode": "deliberate", "needs_clarify_gate": True}) == "clarify_gate"
    assert route_after_goal({"mode": "deliberate", "needs_clarify_gate": False}) == "router"
    assert route_after_goal({"mode": "reason", "needs_clarify_gate": True}) == "reason"
    # heavy тоже проходит через гейт, fast/clarify сюда не доходят (goal только для deliberate/reason/heavy)
    assert route_after_goal({"mode": "heavy", "needs_clarify_gate": True}) == "clarify_gate"
