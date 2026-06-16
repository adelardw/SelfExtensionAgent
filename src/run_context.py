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
