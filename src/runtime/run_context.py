"""
Единый per-request контекст (run_id + user_id) — изоляция процессно-глобального состояния под
КОНКУРЕНТНЫМ сервером (FastAPI `asyncio.create_task`) и Telegram (aiogram).

Выставляется ОДИН раз на ГРАНИЦЕ запроса (`request_scope` в server.py/bot.py вокруг вызова графа),
наследуется ВНИЗ по asyncio-задачам — НЕ внутри ноды (`.set()` в ноде не виден сёстрам, см.
runbudget). Модули резолвят `current_run_id()`/`current_user_id()` и держат per-request состояние в
словарях по ключу (а не в голых глобалах) → два одновременных клиента не затирают друг другу ledger
уточнений, анти-тайпсквоттинг-домены, счётчики, гранты HITL. clarify/interaction УЖЕ изолированы
своими contextvar — здесь не дублируются.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Optional

_run_id: ContextVar[Optional[str]] = ContextVar("sea_run_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("sea_user_id", default=None)
_cleanups: list[Callable[[str], None]] = []


def current_run_id() -> Optional[str]:
    return _run_id.get()


def current_user_id() -> Optional[str]:
    return _user_id.get()


def register_cleanup(fn: Callable[[str], None]) -> None:
    """Модуль регистрирует очистку своего per-run состояния (зовётся по выходе из request_scope)."""
    _cleanups.append(fn)


# ── TAINT: в прогон попал НЕДОВЕРЕННЫЙ внешний контент (веб/документ/чужой репо/MCP) ──────────
# Нужен для гейта python_exec: инжектнутый из такого контента вызов на macOS (rlimits-only) мог бы
# читать ФС и слать наружу → требуем HITL-подтверждение (кроме полного auto). Скоуп — по run_id.
_tainted: dict[str, bool] = {}


def mark_external_content() -> None:
    """Пометить текущий прогон: обработан недоверенный внешний контент."""
    _tainted[current_run_id() or "_default"] = True


def external_content_seen() -> bool:
    """Был ли в прогоне недоверенный внешний контент (веб/док/репо/MCP)."""
    return _tainted.get(current_run_id() or "_default", False)


register_cleanup(lambda rid: _tainted.pop(rid, None))


# ── INTRA-NODE трейс: события внутри нод, которых НЕ видно в node-/tool-трейсе ──
# Покрывает (1) дисциплинированный поиск agentic_research (план/под-вопрос/чтение/verify/синтез —
# это chains/urllib, не «тулы») и (2) раунды step_executor (_exec_direct/_exec_compose: где реально
# уходят секунды шага — генерация vs тул-раунды vs ретраи). Эмитим сюда → сервер кладёт в трейс UI,
# видно где время. Скоуп по run_id (как taint), наследуется вниз по asyncio.
_research: dict[str, list] = {}


def research_emit(text: str) -> None:
    """Событие research-цикла для трейса. Кап, чтобы лог не рос бесконечно при многих под-вопросах."""
    log = _research.setdefault(current_run_id() or "_default", [])
    log.append(text)
    if len(log) > 80:
        del log[: len(log) - 80]


def research_log() -> list:
    return list(_research.get(current_run_id() or "_default", []))


register_cleanup(lambda rid: _research.pop(rid, None))


# ── АРТЕФАКТЫ: файлы, произведённые агентом для ОТДАЧИ пользователю (xlsx/csv/…) ──
# Агент пишет файл нативным тулом (export_table) → регистрирует МЕТА сюда (id/имя/путь). Сервер
# после стрима копирует список в персистентный run-словарь (как research_log) → GUI рисует кнопку
# «скачать» (GET /artifact/{id}); CLI/TUI печатает сохранённый путь. Сами ФАЙЛЫ переживают cleanup
# (лежат в artifacts/<id>/), чистится лишь реестр контекста. Скоуп по run_id.
_artifacts: dict[str, list] = {}


def artifact_emit(meta: dict) -> None:
    """Зарегистрировать произведённый файл (meta: id/name/path/...) для отдачи пользователю."""
    lst = _artifacts.setdefault(current_run_id() or "_default", [])
    # дедуп по id: повторный экспорт того же файла (доп. лист) обновляет запись, не дублирует
    lst[:] = [m for m in lst if m.get("id") != meta.get("id")]
    lst.append(meta)


def artifacts() -> list:
    return list(_artifacts.get(current_run_id() or "_default", []))


register_cleanup(lambda rid: _artifacts.pop(rid, None))


@contextmanager
def request_scope(run_id: str, user_id: str = ""):
    """Изолировать запрос по run_id (+ user_id). Оборачивать вызов графа на границе сервера/бота."""
    t1 = _run_id.set(run_id)
    t2 = _user_id.set(user_id or "")
    try:
        yield
    finally:
        _run_id.reset(t1)
        _user_id.reset(t2)
        for fn in _cleanups:
            try:
                fn(run_id)
            except Exception:  # noqa: BLE001
                pass
