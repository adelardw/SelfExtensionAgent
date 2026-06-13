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
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pydantic import BaseModel

from . import chat_store
from .agent import build_graph, memory_store
from .improve import graph_backward
from .tracing import diagnose, trace_store

app = FastAPI(title="Self-Extension Agent", version="0.2.0")

# GUI — собранный React/Vite-фронт (frontend/dist). Тонкий клиент + мозг: сервер отдаёт
# статику, она говорит с /chat. Собрать: `cd frontend && npm install && npm run build`.
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if (_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
def webui():
    """Веб-GUI на http://localhost:8000/ (или нативное окно desktop.py)."""
    idx = _DIST / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return HTMLResponse(
        "<h1>self-extension</h1><p>GUI не собран. Выполни: "
        "<code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code></p>", status_code=404)
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


# Рабочий буфер связности реплик: последние N сообщений треда (берём из
# персистентного chat_store, поверх долгой памяти). Долгая память остаётся опорой.
_HISTORY_LIMIT = 20


class ChatIn(BaseModel):
    user_id: str
    query: str
    thread_id: str | None = None  # тред-разговор (GUI); по умолчанию = user_id (обратная совместимость)


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
    tid = inp.thread_id or inp.user_id
    cfg = {"configurable": {"thread_id": tid}, "recursion_limit": 50}
    # История треда — из ПЕРСИСТЕНТНОГО chat_store (переживает перезапуск сервера),
    # последние N реплик как рабочий буфер связности.
    history = chat_store.get_messages(tid, last=_HISTORY_LIMIT)
    r = await _graph.ainvoke(
        {"query": inp.query, "user_id": inp.user_id,
         "chat_history": history + [{"role": "user", "content": inp.query}]},
        config=cfg,
    )
    answer = r.get("final_answer", "")
    # Постоянный лог разговора (история тредов в боковой панели GUI / CLI /chats).
    chat_store.record_turn(tid, inp.user_id, inp.query, answer)
    t = chat_store.get_thread(tid)
    return {
        "answer": answer,
        "thread_id": tid,
        "title": (t or {}).get("title", ""),
        "mode": r.get("mode", ""),
        "aim": r.get("aim", ""),
        "confidence": r.get("confidence", 0.0),
        "active_tools": r.get("active_tools", []),
    }


@app.get("/chats")
def chats(user_id: str = "local", limit: int = 50) -> list[dict]:
    """Список тредов-разговоров (избранные первыми, затем по свежести) — боковая панель GUI."""
    return chat_store.list_threads(user_id, limit=limit)


@app.get("/chats/{thread_id}")
def chat_messages(thread_id: str) -> dict:
    """Сообщения одного треда + его метаданные (для открытия разговора в GUI)."""
    return {"thread": chat_store.get_thread(thread_id),
            "messages": chat_store.get_messages(thread_id)}


@app.post("/chats/{thread_id}/favorite")
def chat_favorite(thread_id: str) -> dict:
    """Переключить ★ у треда; возвращает новое состояние."""
    return {"favorite": chat_store.toggle_favorite(thread_id)}


@app.delete("/chats/{thread_id}")
def chat_delete(thread_id: str) -> dict:
    chat_store.delete_thread(thread_id)
    return {"deleted": True}


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
