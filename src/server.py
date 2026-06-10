"""
FastAPI-сервер: единый вход для всех клиентов (ПК, телефон, Telegram).

Запуск:  uvicorn src.server:app --host 0.0.0.0 --port 8000
         (или python -m src.server)

Эндпоинты:
  POST /chat            — основной диалог с агентом
  GET  /diagnose        — самодиагностика (трейсы + деградация)
  GET  /memory/facts    — что агент знает о пользователе (персонализация)
  GET  /memory/goal     — активная стоящая цель + rubric
  GET  /traces          — статистика по нодам графа
"""
from __future__ import annotations

import asyncio
import sqlite3
import time

from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel

from .agent import build_graph, memory_store
from .improve import graph_backward
from .tracing import diagnose, trace_store

app = FastAPI(title="Self-Extension Agent", version="0.2.0")
_cfg = OmegaConf.load("config.yml")
_last_request = time.time()
_idle_done = False  # чтобы не гонять improve повторно за один idle-период

_conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    _graph = build_graph(SqliteSaver(_conn))
except Exception:  # noqa: BLE001
    from langgraph.checkpoint.memory import MemorySaver

    _graph = build_graph(MemorySaver())


# Per-thread диалоговая история в рамках процесса сервера: короткий «рабочий
# буфер» поверх долгой памяти (она остаётся опорой между перезапусками). Ключ —
# user_id (= thread_id). Долгая память персистентна, этот буфер — для связности
# реплик внутри живой сессии.
_CHAT_HISTORY: dict[str, list[dict]] = {}
_HISTORY_LIMIT = 20


class ChatIn(BaseModel):
    user_id: str
    query: str


@app.on_event("startup")
async def _start_idle_loop() -> None:
    """Idle-триггер само-улучшения: после N секунд без запросов запускает graph_backward."""
    imp = _cfg.get("improve", {})

    async def loop():
        global _idle_done
        while True:
            await asyncio.sleep(60)
            idle = time.time() - _last_request
            if imp.get("auto", False) and idle >= imp.get("idle_seconds", 900) and not _idle_done:
                _idle_done = True
                try:
                    res = await asyncio.to_thread(graph_backward, memory_store, imp.get("min_failures", 3))
                    print(f"[idle self-improve] {res}")
                except Exception as e:  # noqa: BLE001
                    print(f"[idle self-improve] failed: {e}")

    asyncio.create_task(loop())


@app.post("/chat")
async def chat(inp: ChatIn) -> dict:
    global _last_request, _idle_done
    _last_request = time.time()
    _idle_done = False
    cfg = {"configurable": {"thread_id": inp.user_id}, "recursion_limit": 50}
    history = _CHAT_HISTORY.get(inp.user_id, [])
    r = await _graph.ainvoke(
        {"query": inp.query, "user_id": inp.user_id,
         "chat_history": history + [{"role": "user", "content": inp.query}]},
        config=cfg,
    )
    answer = r.get("final_answer", "")
    # обновляем рабочий буфер истории этого треда (с обрезкой)
    _CHAT_HISTORY[inp.user_id] = (history + [
        {"role": "user", "content": inp.query},
        {"role": "assistant", "content": answer},
    ])[-_HISTORY_LIMIT:]
    return {
        "answer": answer,
        "mode": r.get("mode", ""),
        "aim": r.get("aim", ""),
        "confidence": r.get("confidence", 0.0),
        "active_tools": r.get("active_tools", []),
    }


@app.get("/diagnose")
def diag(user_id: str = "local") -> dict:
    return diagnose(memory_store, user_id)


@app.get("/memory/facts")
def facts(user_id: str = "local") -> list[dict]:
    return [dict(f) for f in memory_store.get_facts(user_id)]


@app.get("/memory/goal")
def goal(user_id: str = "local") -> dict:
    g = memory_store.get_active_goal(user_id)
    if not g:
        return {"active": False}
    return {"active": True, "aim": g["aim"], "criteria": memory_store.goal_criteria(g)}


@app.get("/traces")
def traces(hours: float = 24.0) -> list[dict]:
    return [dict(s) for s in trace_store.node_stats(hours)]


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
