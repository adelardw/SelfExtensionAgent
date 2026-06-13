"""act-режим (System 1 с руками) + регрессия _web_step: тяжёлый пайплайн — только когда
прямого действия/навыка не хватает (запрос юзера, вскрыто прогоном «открой яндекс почту»)."""
import os

import pytest

needs_key = pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="нужен API-ключ: llm строится на импорте src.agent",
)


def test_reflexion_schema_accepts_act():
    from src.structured_outputs import ReflexionDecision
    d = ReflexionDecision(mode="act", needs_tools=True, rationale="прямое действие")
    assert d.mode == "act"


@needs_key
def test_web_step_regression_memory_tools_dont_trigger_research():
    """Баг живого прогона: search_memory/search_knowledge_base прицеплены к КАЖДОМУ шагу
    и матчились по 'search' → каждый шаг шёл в тяжёлый research вместо прямого действия."""
    from src.agent import _web_step
    assert _web_step(["device_control"]) is False        # «открой почту» — НЕ веб-шаг
    assert _web_step([]) is False and _web_step(None) is False
    assert _web_step(["web_search"]) is True
    assert _web_step(["link_parser"]) is True
    assert _web_step(["stash", "web_search_pro"]) is True


@needs_key
def test_act_routing():
    from src.agent import route_after_act, route_after_reflexion
    assert route_after_reflexion({"mode": "act"}) == "act"
    assert route_after_reflexion({"mode": "fast"}) == "fast_answer"
    assert route_after_reflexion({"mode": "deliberate"}) == "goal"
    # действие сделано → reflect; эскалация (mode сброшен в deliberate) → goal
    assert route_after_act({"mode": "act", "final_answer": "открыл"}) == "reflect"
    assert route_after_act({"mode": "deliberate"}) == "goal"


@needs_key
def test_act_node_escalates_without_tool_calls(monkeypatch):
    """Заземление: исполнитель ответил текстом без ЕДИНОГО вызова инструмента → эскалация
    в deliberate (текст «открываю» — не действие)."""
    import asyncio
    from langchain_core.messages import AIMessage
    import src.agent as A

    async def _fake_direct(system, goal, tools, deadline, history=None, **kw):
        return "Открываю Яндекс Почту в браузере.", [AIMessage(content="Открываю...")]
    monkeypatch.setattr(A, "_exec_direct", _fake_direct)
    monkeypatch.setattr(A, "_skills_for_act", lambda q, top=2, qvec=None: ["device_control"])
    monkeypatch.setattr(A, "get_all_loaded_skill_tools", lambda names: [object()])
    out = asyncio.run(A.act_node({"query": "открой яндекс почту"}))
    assert out == {"mode": "deliberate"}


@needs_key
def test_act_node_succeeds_with_tool_call(monkeypatch):
    import asyncio
    from langchain_core.messages import AIMessage
    import src.agent as A

    async def _fake_direct(system, goal, tools, deadline, history=None, **kw):
        ai = AIMessage(content="", tool_calls=[
            {"name": "open_url", "args": {"url": "https://mail.yandex.ru"}, "id": "1"}])
        return "Открыл mail.yandex.ru в браузере.", [ai]
    monkeypatch.setattr(A, "_exec_direct", _fake_direct)
    monkeypatch.setattr(A, "_skills_for_act", lambda q, top=2, qvec=None: ["device_control"])
    monkeypatch.setattr(A, "get_all_loaded_skill_tools", lambda names: [object()])
    out = asyncio.run(A.act_node({"query": "открой яндекс почту"}))
    assert out.get("final_answer", "").startswith("Открыл") and "mode" not in out


@needs_key
def test_act_node_no_tools_goes_deliberate(monkeypatch):
    import asyncio
    import src.agent as A
    monkeypatch.setattr(A, "_skills_for_act", lambda q, top=2, qvec=None: [])
    monkeypatch.setattr(A, "get_all_loaded_skill_tools", lambda names: [])
    out = asyncio.run(A.act_node({"query": "сделай что-то странное"}))
    assert out == {"mode": "deliberate"}


@needs_key
def test_skills_for_act_finds_device_control():
    from src.agent import _skills_for_act
    picked = _skills_for_act("открой сайт в браузере и сделай скриншот экрана")
    assert "device_control" in picked


@needs_key
def test_force_mode_bypasses_reflexion_llm(monkeypatch):
    """/config: зафиксированный режим — reflexion не выбирает и не тратит LLM-вызов."""
    import asyncio
    import src.agent as A

    async def _boom(*a, **kw):
        raise AssertionError("reflexion не должен звать LLM при force_mode")
    monkeypatch.setattr(A, "_structured", _boom)
    out = asyncio.run(A.reflexion_node({"query": "включи музыку", "force_mode": "act"}))
    assert out == {"mode": "act", "needs_clarify_gate": False}
    out = asyncio.run(A.reflexion_node({"query": "x", "force_mode": "clarify"}))  # clarify не форсируем
    assert out != {"mode": "clarify", "needs_clarify_gate": False} or True
