"""
Память-как-TOOL: агент САМ решает, когда подтянуть память (видение юзера «память
превращается в инструмент»), а не только авто-впрыск memory_context в recall_node.

  • search_memory(query)  — компактная РЕЛЕВАНТНАЯ память (факты/выводы/эпизоды-индекс);
  • recall_history(query) — DRILL-BACK: восстановить ПОЛНЫЕ прошлые эпизоды (вопрос+ответ).
    Компактный recall/саммари = ИНДЕКС; этот тул = доступ к полной истории по запросу
    (как юзер описал: top-k → восстановить полные сообщения).

Изоляция: тулы привязаны к user_id текущего прогона (фабрика на каждый шаг).
"""
from __future__ import annotations

import time

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .memory import MemoryStore


class _MemQuery(BaseModel):
    query: str = Field(description="Что ищем в памяти: тема, факт или прошлый запрос")


class _Note(BaseModel):
    note: str = Field(description="Короткая заметка/промежуточный вывод для текущей задачи")


# ВРЕМЕННЫЙ ярус (runtime scratch): живёт только в рамках прогона, НЕ персистится после
# сессии (видение юзера: «временная — что нужно агенту в RunTime»). Сбрасывается в
# recall_node на старте запроса (clear_scratch). Ключ — user_id.
_SCRATCH: dict[str, list[str]] = {}


def clear_scratch(user_id: str) -> None:
    _SCRATCH.pop(user_id or "default", None)


def make_memory_tools(store: MemoryStore, user_id: str) -> list:
    """Три яруса памяти КАК ТУЛЫ (агент сам решает, что подключить): глобальная (долгая),
    drill-back (полная история), временная (runtime scratch)."""
    uid = user_id or "default"

    def _scratch_write(note: str) -> str:
        _SCRATCH.setdefault(uid, []).append(note.strip())
        return f"Записал во временную память (заметок: {len(_SCRATCH[uid])})."

    def _scratch_read(query: str = "") -> str:
        notes = _SCRATCH.get(uid, [])
        if not notes:
            return "Временная память пуста."
        return "Заметки текущей задачи:\n" + "\n".join(f"- {n}" for n in notes)

    def _search(query: str) -> str:
        try:
            return store.recall(user_id, query, k=5)
        except Exception as e:  # noqa: BLE001
            return f"(память недоступна: {e})"

    def _history(query: str) -> str:
        try:
            eps = store._rank_episodes(user_id, query, 3)  # noqa: SLF001
        except Exception as e:  # noqa: BLE001
            return f"(история недоступна: {e})"
        if not eps:
            return "В истории ничего похожего не найдено."
        out = []
        for ep in eps:
            when = time.strftime("%Y-%m-%d", time.localtime(ep["ts"]))
            out.append(f"[{when}] Запрос: {ep['query']}\nОтвет: {ep['answer']}")
        return "\n\n---\n\n".join(out)

    return [
        # ── ГЛОБАЛЬНЫЙ ярус (долгая память) ──
        StructuredTool.from_function(
            func=_search, name="search_memory", args_schema=_MemQuery,
            description="GLOBAL long-term memory: facts about the user, past conclusions, similar "
                        "past tasks. Use when you need personal/past context you don't currently have.",
        ),
        StructuredTool.from_function(
            func=_history, name="recall_history", args_schema=_MemQuery,
            description="DRILL-BACK to the FULL text of relevant past conversations (the compact "
                        "memory is just an index). Use when the user refers to something said earlier "
                        "or you need the exact prior wording/answer.",
        ),
        # ── ВРЕМЕННЫЙ ярус (runtime scratch, не персистится) ──
        StructuredTool.from_function(
            func=_scratch_write, name="note_to_self", args_schema=_Note,
            description="TEMPORARY working memory: jot an intermediate finding/decision for THIS task "
                        "(not persisted after the session). Use to track progress across steps.",
        ),
        StructuredTool.from_function(
            func=_scratch_read, name="read_my_notes", args_schema=_MemQuery,
            description="Read back your TEMPORARY working notes for the current task.",
        ),
    ]
