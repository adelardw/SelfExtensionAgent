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

import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pydantic import BaseModel

from . import chat_store, cli_config, hitl, knowledge_base, llm, media
from .agent import build_graph, memory_store, rebuild_llms
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
_graph = None       # строится на старте с АСИНХРОННЫМ чекпойнтером


@app.on_event("startup")
async def _build_graph_async() -> None:
    """Граф — с АСИНХРОННЫМ SqliteSaver: /chat зовёт `await ainvoke`, а синхронный
    SqliteSaver async-методы не поддерживает (был источник 500). Фолбэк — MemorySaver
    (состояние графа не персистится, но история тредов всё равно живёт в chat_store)."""
    global _graph
    # GUI: пользователь — локальный оператор, рядом. Без auto-accept side-effect тулы
    # уходят в deny-by-default (нет канала подтверждения в окне) → агент «ждёт
    # подтверждения», которое не появляется. auto-accept → браузер/плеер реально работают.
    # (Коммерцию/покупки агент всё равно отказывается делать на уровне логики.)
    hitl.set_work_mode(cli_config.get_cli("work_mode") or "auto-accept")
    Path("data").mkdir(parents=True, exist_ok=True)
    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect("data/checkpoints.db")
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        _graph = build_graph(saver)
    except Exception as e:  # noqa: BLE001
        from langgraph.checkpoint.memory import MemorySaver

        print(f"[server] AsyncSqliteSaver недоступен ({type(e).__name__}: {e}) → MemorySaver")
        _graph = build_graph(MemorySaver())


# Рабочий буфер связности реплик: последние N сообщений треда (берём из
# персистентного chat_store, поверх долгой памяти). Долгая память остаётся опорой.
_HISTORY_LIMIT = 20
_THREAD_FILES: dict[str, list[str]] = {}  # thread_id → приложенные файлы (сессионные вложения)


class ChatIn(BaseModel):
    user_id: str
    query: str
    thread_id: str | None = None  # тред-разговор (GUI); по умолчанию = user_id (обратная совместимость)


class SettingsIn(BaseModel):
    provider: str | None = None      # "openrouter" | "ollama"
    model: str | None = None         # основная (fast) модель
    code_model: str | None = None    # модель для кода/исполнения
    deep_model: str | None = None    # модель для тяжёлого ревью
    base_url: str | None = None
    api_key: str | None = None
    work_mode: str | None = None     # "manual" | "auto-accept" | "auto"
    force_mode: str | None = None    # "" (авто) | fast | reason | act | deliberate | heavy


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
    # Приложенные к треду файлы → подмешиваем их содержимое/описание в запрос.
    query = inp.query
    files = _THREAD_FILES.get(tid)
    if files:
        try:
            ctx = await asyncio.to_thread(media.attachment_context, files, inp.query)
            if ctx:
                query = f"{inp.query}\n\n=== ПРИЛОЖЕННЫЕ ФАЙЛЫ ===\n{ctx}"
        except Exception as e:  # noqa: BLE001
            print(f"[chat] attachment_context failed: {e}")
    r = await _graph.ainvoke(
        {"query": query, "user_id": inp.user_id, "session_id": tid,
         "force_mode": cli_config.get_cli("force_mode") or "",
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


@app.post("/upload")
async def upload(thread_id: str, file: UploadFile = File(...)) -> dict:
    """Приложить документ/файл к треду (pdf/doc/image/audio/…) — мультимодальное вложение."""
    suffix = "".join(c for c in (file.filename or "file") if c.isalnum() or c in "._- ")
    tmp = Path(tempfile.gettempdir()) / f"selfext_{thread_id}_{suffix}"
    tmp.write_bytes(await file.read())
    try:
        stored = await asyncio.to_thread(knowledge_base.add_session_file, thread_id, str(tmp))
    except Exception:  # noqa: BLE001
        stored = str(tmp)
    _THREAD_FILES.setdefault(thread_id, []).append(stored)
    return {"name": file.filename, "count": len(_THREAD_FILES[thread_id])}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    """Запись с микрофона → текст (STT). Фронт пишет аудио и шлёт сюда."""
    ext = Path(file.filename or "audio.webm").suffix or ".webm"
    tmp = Path(tempfile.gettempdir()) / f"selfext_rec{ext}"
    tmp.write_bytes(await file.read())
    try:
        text = await asyncio.to_thread(media.transcribe_audio, str(tmp))
        return {"text": text or ""}
    except Exception as e:  # noqa: BLE001
        return {"text": "", "error": f"{type(e).__name__}: {e}"}


@app.get("/settings")
def get_settings() -> dict:
    """Текущие настройки для панели GUI (ключ не раскрываем — только источник)."""
    return {
        "provider": cli_config.get_cli("provider") or _cfg.get("provider", "openrouter"),
        "model": llm.model_for("fast"),
        "code_model": llm.model_for("code"),
        "deep_model": llm.model_for("deep"),
        "base_url": cli_config.get_cli("base_url") or "",
        "api_key_source": llm.api_key_source(),
        "work_mode": hitl.work_mode(),
        "force_mode": cli_config.get_cli("force_mode") or "",
        "active": llm.active_summary(),
    }


@app.post("/settings")
def post_settings(s: SettingsIn) -> dict:
    """Сохранить настройки (persist в config.local.yml) + живая валидация ключа."""
    for key, val in (("provider", s.provider), ("model", s.model), ("code_model", s.code_model),
                     ("deep_model", s.deep_model), ("force_mode", s.force_mode)):
        if val is not None:
            cli_config.set_cli(key, val or None)
    if s.base_url is not None:
        cli_config.set_cli("base_url", s.base_url or None)
    if s.api_key is not None:
        cli_config.set_cli("api_key", s.api_key or None)
    if s.work_mode is not None:
        cli_config.set_cli("work_mode", s.work_mode)
        hitl.set_work_mode(s.work_mode)
    llm.set_provider(cli_config.get_cli("provider"), cli_config.get_cli("model"))
    rebuild_llms()
    ok, msg = llm.validate_credentials()
    return {"ok": ok, "message": msg, "active": llm.active_summary(),
            "api_key_source": llm.api_key_source(), "work_mode": hitl.work_mode()}


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
