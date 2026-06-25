"""
FastAPI-сервер: единый вход для всех клиентов (ПК, телефон, Telegram).

Запуск:  python -m src.server   (бэкенд на 127.0.0.1:8000 — loopback по умолчанию)
         LAN-доступ (телефон на том же Wi-Fi) — ОСОЗНАННО: AGENT_BIND_HOST=0.0.0.0
         (тогда нужен auth перед эндпойнтами — сейчас их нет).

Эндпоинты:
  POST /chat            — основной диалог с агентом
  GET  /diagnose        — самодиагностика (трейсы + деградация)
  GET  /memory/facts    — что агент знает о пользователе (персонализация)
  GET  /memory/goal     — активная стоящая цель + rubric
  GET  /traces          — статистика по нодам графа
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from pathlib import Path

import contextvars
import tempfile
import uuid

from langchain_core.callbacks import BaseCallbackHandler

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.browser import browser_bridge
from src.interface import chat_store, clarify
from src.config import cli_config
from src.runtime import hitl, runbudget, run_context
from src.data import knowledge_base
from src.llm import llm, usage
from src import media
from src.graph.agent import build_graph, memory_store, rebuild_llms
from src.improve import graph_backward
from src.tracing import diagnose, trace_store
from src.tools import skill_health, skill_repair

app = FastAPI(title="Self-Extension Agent", version="0.2.0")

# GUI — собранный React/Vite-фронт (frontend/dist). Тонкий клиент + мозг: сервер отдаёт
# статику, она говорит с /chat. Собрать: `cd frontend && npm install && npm run build`.
_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"  # interface→src→root
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
    # SearXNG: в .app cwd=support-каталог → .env не грузится → SEARXNG_URL пуст. Берём из
    # GUI-конфига (задаётся в Настройках) и кладём в окружение, чтобы веб/image-поиск его видел.
    _sx = cli_config.get_cli("searxng_url")
    if _sx:
        import os as _o
        _o.environ["SEARXNG_URL"] = _sx
    # Модальности текущей модели (image/audio/…) — узнаём у OpenRouter раз на старте, кэшируем (фон,
    # не блокируем старт): чтобы картинка шла напрямую без пере-сохранения настроек.
    asyncio.create_task(asyncio.to_thread(llm.remember_modalities))
    # Интерактивные каналы: уточнения (Q/A мультиселект) и подтверждения surface в GUI.
    clarify.set_clarifier(_server_clarifier)
    hitl.set_confirmer(_server_confirmer)
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

    # Мост браузерного расширения — поднимаем СРАЗУ при старте, чтобы расширение могло
    # подключиться (агент в ТВОЁМ Chrome). Идемпотентно. Печатаем токен для попапа.
    try:
        browser_bridge.ensure_server()

        async def _ext_chat(text: str) -> str:
            """Чат ИЗ попапа расширения → тот же граф, тред 'extension'."""
            try:
                hist = chat_store.get_messages("extension", last=_HISTORY_LIMIT)
                with run_context.request_scope(f"ext-{uuid.uuid4().hex}", "local"):  # изоляция per-request
                    r = await _graph.ainvoke(
                        {"query": text, "user_id": "local", "session_id": "extension",
                         "interactive": True,  # desktop/extension: доводим без «продолжи»
                         "chat_history": hist + [{"role": "user", "content": text}]},
                        config={"configurable": {"thread_id": "extension"}, "recursion_limit": 50})
                ans = r.get("final_answer", "") or "(пустой ответ)"
                chat_store.record_turn("extension", "local", text, ans)
                return ans
            except Exception as ex:  # noqa: BLE001
                return f"ошибка: {type(ex).__name__}: {ex}"

        browser_bridge.set_chat_handler(_ext_chat)
        print(f"[server] 🧩 мост браузера слушает 127.0.0.1:{browser_bridge.PORT} · "
              f"токен для расширения: {browser_bridge.token()}")
    except Exception as e:  # noqa: BLE001
        print(f"[server] мост браузера не поднялся: {type(e).__name__}: {e}")


# Рабочий буфер связности реплик: последние N сообщений треда (берём из
# персистентного chat_store, поверх долгой памяти). Долгая память остаётся опорой.
_HISTORY_LIMIT = 20
_THREAD_FILES: dict[str, list[str]] = {}  # thread_id → приложенные файлы (сессионные вложения)

# ── Интерактивные прогоны: уточнения (Q/A с мультиселектом) и подтверждения по HTTP ──
# Прогон идёт фоновой задачей; когда граф зовёт clarify/confirm — кладём «pending» в реестр
# и ЖДЁМ Future. Клиент поллит /run/{id}, рендерит Q/A, шлёт ответ в /run/{id}/respond.
_RUNS: dict[str, dict] = {}
_cur_run: contextvars.ContextVar[str] = contextvars.ContextVar("server_run", default="")


async def _await_user(kind: str, payload: dict, timeout: float = 900.0):
    """Выставить pending текущему прогону и ждать ответ клиента (или None по таймауту)."""
    run = _RUNS.get(_cur_run.get())
    if not run:
        return None
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    run["pending"] = {"type": kind, **payload}
    run["future"] = fut
    run["status"] = "waiting"
    try:
        ans = await asyncio.wait_for(fut, timeout=timeout)
    except Exception:  # noqa: BLE001 — таймаут/отмена → допущение
        ans = None
    run["pending"] = None
    run["future"] = None
    run["status"] = "running"
    return ans


async def _server_clarifier(items: list[dict]):
    """Канал уточнений: показать вопросы (с вариантами) в GUI, вернуть ответы."""
    qs = [{"question": it.get("question", ""), "options": list(it.get("options", []) or []),
           "why": it.get("why", "")} for it in items]
    return await _await_user("clarify", {"questions": qs})  # list[str] | None


async def _server_confirmer(description: str):
    """Канал подтверждений (активен в режиме manual; в auto-accept не зовётся)."""
    ans = await _await_user("confirm", {"text": description})
    return ans if ans is not None else False


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
    searxng_url: str | None = None   # свой SearXNG для веб/image-поиска (напр. http://localhost:8080)


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


class RespondIn(BaseModel):
    answers: list[str] | None = None   # ответы на clarify-вопросы (по порядку)
    value: str | None = None           # ответ на confirm


# Узлы графа → понятные шаги для GUI (видимый ход исполнения).
_PROGRESS = {
    "recall": "Recalling context", "goal": "Setting the goal", "reflexion": "Choosing the approach",
    "act": "Acting", "reason": "Reasoning", "fast_answer": "Answering", "clarify_gate": "Clarifying",
    "router": "Routing", "create_skills": "Creating a skill", "sgr_create": "Validating the skill",
    "skill_selector": "Selecting skills", "capability_research": "Finding capability / MCP",
    "decompose": "Breaking into steps", "skill_injection": "Wiring up tools",
    "step_executor": "Executing a step", "synthesize": "Assembling the answer", "review": "Deep review",
    "validation": "Validating", "reflect": "Remembering",
}


class _TraceCb(BaseCallbackHandler):
    """Лёгкий трейс для интерпретируемости в UI: ТУЛЫ, которые агент реально звал, по порядку
    (включая MCP-тулы), с РАНТАЙМОМ каждого (on_tool_end − on_tool_start) — видно, что внутри шага
    ело время (search_web / browse / LLM-раунды). Узлы графа идут отдельно (run['steps'])."""

    def __init__(self) -> None:
        self.tools: list[dict] = []        # [{"name": str, "sec": float|None}] по порядку вызова
        self._starts: dict[str, tuple] = {}  # run_id → (idx, monotonic-старт)

    def on_tool_start(self, serialized, input_str=None, *, run_id=None, **kwargs) -> None:  # noqa: ANN001, ARG002
        name = (serialized or {}).get("name") if isinstance(serialized, dict) else ""
        name = name or kwargs.get("name") or "tool"
        self.tools.append({"name": name, "sec": None})
        if run_id is not None:
            self._starts[str(run_id)] = (len(self.tools) - 1, time.monotonic())

    def on_tool_end(self, output, *, run_id=None, **kwargs) -> None:  # noqa: ANN001, ARG002
        st = self._starts.pop(str(run_id), None)
        if st:
            idx, t0 = st
            if 0 <= idx < len(self.tools):
                self.tools[idx]["sec"] = round(time.monotonic() - t0, 1)


async def _heal_bg() -> None:
    """Фоновая самопочинка деградировавших навыков после прогона (best-effort, не трогает ответ)."""
    try:
        res = await skill_repair.heal_degraded()
        for r in res:
            print(f"[self-heal] {r.get('skill')}: {'✓' if r.get('ok') else '✗'} {r.get('reason')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[self-heal] фон упал: {type(e).__name__}: {e}", flush=True)


async def _run_graph(run_id: str, inp: ChatIn, tid: str) -> None:
    """Фоновый прогон через astream: прогресс по узлам → run['progress'], а clarify/confirm
    всплывают в _RUNS[run_id] и ждут клиента."""
    _cur_run.set(run_id)
    run = _RUNS[run_id]
    try:
        query = inp.query
        files = _THREAD_FILES.get(tid)
        # САНИТАЙЗ: add_session_file при несуществующем src возвращает СТРОКУ "[файл не найден: …]",
        # которая попадала в _THREAD_FILES как «путь» → потом «не найден» + конфабуляция. Отсекаем
        # такие записи и реально несуществующие пути ДО обработки.
        if files:
            files = [f for f in files if not str(f).startswith("[") and Path(f).exists()]
        # Диагностика в трейс: что РЕАЛЬНО дошло (имена + существование) — чтобы не гадать.
        run["attach"] = [{"n": Path(f).name, "ok": Path(f).exists()}
                         for f in (_THREAD_FILES.get(tid) or [])]
        image_paths: list = []
        if files:
            _att_t0 = time.monotonic()  # «Reading attachments» — РЕАЛЬНЫЙ шаг в трейсе (с таймингом)
            run["progress"] = "Reading attachments"
            if "Reading attachments" not in run["steps"]:
                run["steps"].append("Reading attachments")
            imgs = [f for f in files if Path(f).suffix.lower() in media.IMAGE_EXTS]
            run["n_attached"], run["n_images"] = len(files), len(imgs)  # для трейса: дошло ли вложение
            # vision_supported при непрогретом кэше делает сетевой фетч (urlopen 8с) → в поток, чтобы
            # НЕ морозить event loop сервера (после первого раза — мгновенно из кэша).
            direct = bool(imgs) and await asyncio.to_thread(llm.vision_supported)
            image_paths = imgs if direct else []
            # direct (мультимодальная модель) → fast/reason читают картинку САМИ (image_url), описывать
            # её НЕ надо (это были лишние ~10с/картинку vision-вызовов на «Reading attachments»).
            # НЕ direct (text-only) → описываем в текст (единственный способ её «прочитать»).
            others = [f for f in files if f not in imgs] if direct else list(files)
            if image_paths:  # пометка для reflexion/decompose (они видят только текст)
                query = f"{query}\n\n[К сообщению приложено изображений: {len(image_paths)} — посмотри на них.]"
            if others:
                try:
                    ctx = await media.attachment_context_async(others, inp.query)
                    if ctx:
                        query = f"{query}\n\n=== ПРИЛОЖЕННЫЕ ФАЙЛЫ ===\n{ctx}"
                except Exception as e:  # noqa: BLE001
                    print(f"[chat] attachment_context failed: {e}")
            run.setdefault("timings", {})["Reading attachments"] = round(time.monotonic() - _att_t0, 1)
        history = chat_store.get_messages(tid, last=_HISTORY_LIMIT)
        state: dict = {}
        _scope_id = f"chat-{uuid.uuid4().hex}"
        with run_context.request_scope(_scope_id, inp.user_id):  # изоляция per-request
            tracker = usage.TokenTracker()  # ловит usage каждого LLM-вызова прогона (в десктопе их не было)
            trace_cb = _TraceCb()           # трейс тулов для интерпретируемости в UI
            _last_ts = time.monotonic()     # для тайминга по нодам (где реально уходит время)
            async for chunk in _graph.astream(
                {"query": query, "user_id": inp.user_id, "session_id": tid,
                 "force_mode": cli_config.get_cli("force_mode") or "",
                 "image_paths": image_paths,  # картинки — напрямую в мультимодальную модель
                 "has_attachments": bool(files),  # свежие вложения → recall не подменяет их findings'ами
                 "interactive": True,  # desktop: доводим план без «продолжи» (план самоограничен)
                 "chat_history": history + [{"role": "user", "content": inp.query}]},
                config={"configurable": {"thread_id": tid}, "recursion_limit": 50,
                        "callbacks": [tracker, trace_cb]},
                stream_mode="updates",
            ):
                for node, delta in (chunk or {}).items():
                    _now = time.monotonic()
                    lbl = _PROGRESS.get(node, node)
                    # длительность ноды ≈ время между апдейтами (update приходит, когда нода завершилась).
                    # Накапливаем по метке (шаги executor повторяются) → видно, ЧТО ел время.
                    run.setdefault("timings", {})[lbl] = round(
                        run.get("timings", {}).get(lbl, 0.0) + (_now - _last_ts), 1)
                    _last_ts = _now
                    run["progress"] = lbl
                    if lbl not in run["steps"]:
                        run["steps"].append(lbl)
                    if isinstance(delta, dict):
                        state.update(delta)
                    run["tokens"] = tracker.total       # живой счётчик на лету (для поллинга)
                    run["tools_used"] = list(trace_cb.tools)            # трейс: какие тулы звал
                    run["mode"] = state.get("mode", "") or run.get("mode", "")
                    run["rationale"] = state.get("mode_rationale", "") or run.get("rationale", "")
                    run["findings_used"] = bool(state.get("findings_used"))  # сработал ли кэш находок
                    rlog = run_context.research_log()  # шаги дисциплинированного research (intra-node)
                    if rlog:
                        run["research_log"] = rlog
                    arts = run_context.artifacts()  # файлы, произведённые агентом для отдачи
                    if arts:  # наружу — без серверного path (утечка ФС), только id/name/nrows/kind
                        run["artifacts"] = [{k: a.get(k) for k in ("id", "name", "nrows", "kind")} for a in arts]
        answer = state.get("final_answer", "")
        chat_store.record_turn(tid, inp.user_id, inp.query, answer)
        t = chat_store.get_thread(tid)
        run["result"] = {"answer": answer, "thread_id": tid, "title": (t or {}).get("title", ""),
                         "mode": state.get("mode", ""), "active_tools": state.get("active_tools", []),
                         "tools_used": list(trace_cb.tools), "rationale": state.get("mode_rationale", ""),
                         "findings_used": bool(state.get("findings_used")), "timings": dict(run.get("timings", {})),
                         "research_log": run.get("research_log", []),
                         "artifacts": run.get("artifacts", []),
                         "n_attached": run.get("n_attached", 0), "n_images": run.get("n_images", 0),
                         "attach": run.get("attach", []),
                         "tokens": tracker.total, "tokens_in": tracker.input, "tokens_out": tracker.output,
                         "cost": round(tracker.cost(), 4), "cached_rate": round(tracker.cache_hit_rate, 2)}
        run["status"] = "done"
        # САМОПОЧИНКА навыков В ФОНЕ (не в графе → нет user-latency): если за прогон навык
        # деградировал (серия сбоев одного класса), чиним после ответа. Loop сервера живёт → задача
        # доработает. Ошибки самопочинки не трогают ответ (фон, best-effort).
        if skill_health.degraded():
            asyncio.create_task(_heal_bg())
    except Exception as e:  # noqa: BLE001
        run["error"] = f"{type(e).__name__}: {e}"
        run["status"] = "error"
    finally:
        # Вложения — ПЕР-СООБЩЕНИЕ: чистим после прогона, иначе _THREAD_FILES копит ВСЕ картинки треда
        # → следующий запрос обрабатывает старые (часть удалена → «не считаны») и агент конфабулирует
        # из findings. UI и так гасит чипы при отправке — синхронизируем бэкенд. (clarify — внутри astream,
        # сюда доходим только по завершении всего прогона.)
        _THREAD_FILES.pop(tid, None)


@app.post("/chat")
async def chat(inp: ChatIn) -> dict:
    """Старт прогона. Возвращает run_id; клиент поллит /run/{run_id}."""
    global _last_request, _idle_done
    _last_request = time.time()
    _idle_done = False
    tid = inp.thread_id or inp.user_id
    run_id = uuid.uuid4().hex
    _RUNS[run_id] = {"status": "running", "pending": None, "future": None, "result": None,
                     "error": None, "progress": "Starting", "steps": []}
    if len(_RUNS) > 60:  # анти-утечка: чистим старые завершённые
        for k in [k for k, v in list(_RUNS.items()) if v["status"] in ("done", "error")][:40]:
            _RUNS.pop(k, None)
    asyncio.create_task(_run_graph(run_id, inp, tid))
    return {"run_id": run_id}


@app.get("/run/{run_id}")
def run_status(run_id: str) -> dict:
    """Статус прогона: running | waiting(+pending Q/A) | done(+answer) | error."""
    run = _RUNS.get(run_id)
    if not run:
        return {"status": "unknown"}
    st = run["status"]
    _trace = {"tokens": run.get("tokens", 0), "tools_used": run.get("tools_used", []),
              "mode": run.get("mode", ""), "rationale": run.get("rationale", ""),
              "findings_used": run.get("findings_used", False), "timings": run.get("timings", {}),
              "research_log": run.get("research_log", []),
              "artifacts": run.get("artifacts", []),
              "n_attached": run.get("n_attached", 0), "n_images": run.get("n_images", 0),
              "attach": run.get("attach", [])}
    if st == "waiting":
        return {"status": "waiting", "pending": run["pending"], "progress": run["progress"],
                "steps": run["steps"], **_trace}
    if st == "done":
        return {"status": "done", **(run["result"] or {}), "steps": run["steps"]}
    if st == "error":
        return {"status": "error", "error": run["error"], "steps": run["steps"]}
    return {"status": "running", "progress": run["progress"], "steps": run["steps"], **_trace}


@app.get("/artifact/{artifact_id}")
def get_artifact(artifact_id: str):
    """Отдать произведённый агентом файл (xlsx/csv) на скачивание. id жёстко валидируется внутри
    resolve_artifact (только hex, путь строго в artifacts/<id>/ → анти-traversal)."""
    from src.runtime.artifacts import resolve_artifact
    from fastapi import HTTPException
    p = resolve_artifact(artifact_id)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(str(p), filename=p.name, media_type="application/octet-stream")


@app.post("/run/{run_id}/respond")
async def run_respond(run_id: str, r: RespondIn) -> dict:
    """Ответ пользователя на pending уточнение/подтверждение → продолжить прогон."""
    run = _RUNS.get(run_id)
    fut = run and run.get("future")
    if not fut or fut.done():
        return {"ok": False}
    if (run["pending"] or {}).get("type") == "clarify":
        fut.set_result(r.answers or [])
    else:
        fut.set_result(r.value if r.value is not None else "да")
    return {"ok": True}


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
    lst = _THREAD_FILES.setdefault(thread_id, [])
    if stored not in lst:
        lst.append(stored)
    return {"name": file.filename, "count": len(_THREAD_FILES[thread_id]),
            "vision_warn": await _vision_warn([file.filename or ""])}


class AttachLocalIn(BaseModel):
    thread_id: str
    paths: list[str]


@app.post("/attach_local")
async def attach_local(inp: AttachLocalIn) -> dict:
    """Десктоп (pywebview): нативный файл-диалог отдаёт ЛОКАЛЬНЫЕ пути, а сервер на той же машине —
    читает их сам (без заливки контента по HTTP). В WKWebView программный клик по <input type=file>
    не открывает панель → фронт зовёт pywebview.api.pick_files() и шлёт пути сюда."""
    names: list[str] = []
    errors: list[str] = []
    for p in inp.paths:
        try:
            stored = await asyncio.to_thread(knowledge_base.add_session_file, inp.thread_id, p)
            lst = _THREAD_FILES.setdefault(inp.thread_id, [])
            if stored not in lst:  # дедуп: не плодим дубли (был баг «5 вместо 3»)
                lst.append(stored)
            names.append(Path(p).name)
        except BaseException as e:  # noqa: BLE001 — pyo3-панику парсеров тоже ловим, не 500
            errors.append(f"{Path(p).name}: {type(e).__name__}")
    return {"names": names, "errors": errors, "count": len(_THREAD_FILES.get(inp.thread_id, [])),
            "vision_warn": await _vision_warn(inp.paths)}


async def _vision_warn(names: list) -> bool:
    """Среди вложений есть картинка, а текущая модель её НЕ читает (text-only)? Тогда предупреждаем
    юзера выбрать мультимодальную модель — описать картинку всё равно нечем (vision идёт через ту же
    модель). vision_supported читает кэш модальностей (фетч раз на модель) → в поток, не морозим loop."""
    if not any(Path(n).suffix.lower() in media.IMAGE_EXTS for n in (names or [])):
        return False
    return not await asyncio.to_thread(llm.vision_supported)


class DetachIn(BaseModel):
    thread_id: str
    name: str  # имя файла (basename), как показано в чипе


@app.post("/detach")
async def detach(inp: DetachIn) -> dict:
    """Убрать ранее приложенный файл из вложений треда (по имени). Чистим in-memory список (его
    читает прогон) и удаляем копию из session-стора. Идемпотентно: нет файла → просто count."""
    files = _THREAD_FILES.get(inp.thread_id, [])
    kept, removed = [], []
    for f in files:
        (removed if Path(f).name == inp.name else kept).append(f)
    _THREAD_FILES[inp.thread_id] = kept
    for f in removed:
        try:
            Path(f).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — копия в tmp/session, удалить best-effort
            pass
    return {"removed": [Path(f).name for f in removed], "count": len(kept)}


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


# Десктоп (pywebview/WKWebView): getUserMedia НЕ поддерживается → пишем с мика на СЕРВЕРЕ через ffmpeg
# (avfoundation), как в TUI. start → запись, stop → стоп + расшифровка. Состояние по thread_id.
_REC: dict = {}


def _end_proc(proc) -> None:
    """Гарантированно завершить процесс записи и ОТПУСТИТЬ микрофон: SIGTERM → ждём чуть →
    SIGKILL, если не умер. Без kill-фолбэка ffmpeg иногда игнорит SIGTERM и держит мик висяком."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:  # noqa: BLE001 — не умер по-хорошему → жёстко
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            pass


def _find_ffmpeg() -> str | None:
    """ffmpeg по PATH, иначе в стандартных местах. ВАЖНО для .app: из Launchpad PATH урезан
    (/usr/bin:/bin:…, без /opt/homebrew/bin) → shutil.which не находит установленный brew-ffmpeg."""
    import os as _os
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    for c in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg",
              str(Path.home() / "homebrew" / "bin" / "ffmpeg")):
        if _os.path.exists(c):
            return c
    return None


@app.on_event("shutdown")
def _release_mic_on_exit() -> None:
    """Закрыли приложение во время записи → отпустить микрофон (иначе ffmpeg-сирота держит мик)."""
    for proc, _wav in list(_REC.values()):
        _end_proc(proc)
    _REC.clear()


@app.post("/voice/start")
async def voice_start(thread_id: str) -> dict:
    import subprocess
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return {"ok": False, "error": "ffmpeg не найден (brew install ffmpeg)"}
    if thread_id in _REC:
        return {"ok": True}  # уже пишем
    wav = Path(tempfile.gettempdir()) / f"selfext_voice_{thread_id}.wav"
    try:
        proc = subprocess.Popen(
            # -t 300: страховка от утечки мика — ffmpeg САМ остановится через 5 мин, даже если
            # /voice/stop не пришёл (закрыли окно/сбой). Иначе процесс держит микрофон бесконечно.
            [ffmpeg, "-y", "-loglevel", "error", "-f", "avfoundation", "-i", ":0",
             "-ac", "1", "-ar", "16000", "-t", "300", str(wav)],
            stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    _REC[thread_id] = (proc, str(wav))
    return {"ok": True}


@app.post("/voice/stop")
async def voice_stop(thread_id: str) -> dict:
    import os as _os
    rec = _REC.pop(thread_id, None)
    if not rec:
        return {"text": ""}
    proc, wav = rec
    _end_proc(proc)  # terminate → wait → SIGKILL: гарантированно отпустить микрофон
    text = ""
    try:
        if _os.path.exists(wav) and _os.path.getsize(wav) > 1200:
            text = await asyncio.to_thread(media.transcribe_audio, wav) or ""
    except Exception as e:  # noqa: BLE001
        return {"text": "", "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            _os.unlink(wav)
        except OSError:
            pass
    return {"text": text}


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
        "searxng_url": cli_config.get_cli("searxng_url") or os.getenv("SEARXNG_URL", ""),
        "model_modalities": llm.model_modalities(),  # входные модальности модели (image/audio/…)
        "active": llm.active_summary(),
        "bridge_connected": browser_bridge.connected(),
        "bridge_token": browser_bridge.token(),
        "bridge_port": browser_bridge.PORT,
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
    if s.searxng_url is not None:
        cli_config.set_cli("searxng_url", s.searxng_url or None)
        os.environ["SEARXNG_URL"] = s.searxng_url or ""  # живо: веб/image-поиск подхватят сразу
    llm.set_provider(cli_config.get_cli("provider"), cli_config.get_cli("model"))
    rebuild_llms()
    # Модель сменилась → один раз спрашиваем у OpenRouter её модальности (image/audio/…) и запоминаем.
    if s.provider is not None or s.model is not None:
        llm.remember_modalities()
    ok, msg = llm.validate_credentials()
    return {"ok": ok, "message": msg, "active": llm.active_summary(),
            "api_key_source": llm.api_key_source(), "work_mode": hitl.work_mode()}


@app.get("/diagnose")
def diag(user_id: str = "local") -> dict:
    d = diagnose(memory_store, user_id)
    d["skill_health"] = skill_health.all_health()       # per-skill вызовы/сбои/статус
    d["skills_degraded"] = skill_health.degraded()      # кандидаты на самопочинку
    return d


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
    import os

    import uvicorn

    # Локальное приложение: бэкенд слушает LOOPBACK по умолчанию (как desktop.py:HOST=127.0.0.1).
    # 0.0.0.0 без auth на недоверенной сети = любой в LAN управляет агентом как владелец
    # (auto-accept + руки/память/ключ). LAN-доступ (телефон на том же Wi-Fi) — ОСОЗНАННЫЙ opt-in
    # через AGENT_BIND_HOST=0.0.0.0 (и тогда добавляй auth перед эндпойнтами). Bind-находка ревью.
    host = os.getenv("AGENT_BIND_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=8000)


if __name__ == "__main__":
    main()
