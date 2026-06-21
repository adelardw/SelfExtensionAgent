"""Thread 4 — маскинг контекста внутри шага: старые ToolMessage сворачиваются, последние
держатся полными; tool_call_id и парность сохраняются (структура не рвётся)."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph.agent import _mask_old_tool_msgs


def _build(n: int) -> list:
    msgs = [HumanMessage(content="задача")]
    for i in range(n):
        msgs.append(AIMessage(content=f"ход {i}"))
        msgs.append(ToolMessage(content="наблюдение " * 50, tool_call_id=f"id{i}"))
    return msgs


def test_keeps_recent_folds_old():
    msgs = _build(8)
    _mask_old_tool_msgs(msgs, keep=3)
    tms = [m for m in msgs if m.__class__.__name__ == "ToolMessage"]
    assert len(tms) == 8
    # старые (все кроме последних 3) — свёрнуты
    assert all("свёрнуто" in m.content for m in tms[:-3])
    # последние 3 — полные
    assert all("свёрнуто" not in m.content for m in tms[-3:])
    # tool_call_id сохранены и в порядке (парность с AIMessage не разрушена)
    assert [m.tool_call_id for m in tms] == [f"id{i}" for i in range(8)]


def test_noop_when_few():
    msgs = _build(2)
    before = [m.content for m in msgs if m.__class__.__name__ == "ToolMessage"]
    _mask_old_tool_msgs(msgs, keep=4)
    after = [m.content for m in msgs if m.__class__.__name__ == "ToolMessage"]
    assert before == after  # меньше keep → ничего не сворачиваем


def test_idempotent():
    msgs = _build(8)
    _mask_old_tool_msgs(msgs, keep=3)
    once = [m.content for m in msgs if m.__class__.__name__ == "ToolMessage"]
    _mask_old_tool_msgs(msgs, keep=3)  # повторный проход не должен сворачивать дважды
    twice = [m.content for m in msgs if m.__class__.__name__ == "ToolMessage"]
    assert once == twice


def test_short_messages_not_folded():
    msgs = [HumanMessage(content="t")]
    for i in range(6):
        msgs.append(AIMessage(content="x"))
        msgs.append(ToolMessage(content="ok", tool_call_id=f"id{i}"))  # короткое (<80) — не трогаем
    _mask_old_tool_msgs(msgs, keep=2)
    tms = [m for m in msgs if m.__class__.__name__ == "ToolMessage"]
    assert all(m.content == "ok" for m in tms)  # короткие наблюдения остаются как есть
