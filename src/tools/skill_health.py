"""
Здоровье навыков во времени (контур самопочинки).

Дыра, которую закрывает: `collective.py` считает winrate РЕЦЕПТОВ (поведенческих паттернов),
`degradation.py` — процессный счётчик fallback'ов. А вот ЗДОРОВЬЯ КОДА отдельного навыка во
времени нет: если у навыка сломался внешний API (изменился endpoint/схема), он падает молча
прогон за прогоном, и никто это не чинит. Здесь — лёгкий per-skill учёт: вызовы/сбои/класс
последней ошибки/сбоев-подряд + АРГУМЕНТЫ последнего падавшего вызова (для регрессии починки).

Персист — atomic JSON в data/ (как registry), под локом (фон-reflect ↔ main-прогон). Скоуп —
процессно-кумулятивный (здоровье навыка — свойство кода, не прогона).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
DEGRADE_AFTER = int(os.getenv("SKILL_DEGRADE_AFTER", "3"))  # N сбоев ОДНОГО класса подряд → degraded


def _path() -> Path:
    d = Path("data")
    d.mkdir(parents=True, exist_ok=True)
    return d / "skill_health.json"


def _load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text("utf-8")) or {}
    except Exception:  # noqa: BLE001 — битый файл не должен ронять прогон
        return {}


def _save(data: dict) -> None:
    p = _path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    os.replace(tmp, p)  # atomic


# Семейства родственных типов исключений (timeout/network/import/http) — чтобы напр. ReadTimeout и
# ConnectTimeout считались ОДНИМ классом «сервис тормозит». Всё прочее группируется по САМОМУ типу
# исключения (KeyError≠ValueError) — это структурная идентичность сбоя, НЕ подстрочное гадание по тексту.
_TYPE_FAMILY = {
    "TimeoutError": "timeout", "ReadTimeout": "timeout", "ConnectTimeout": "timeout", "timeout": "timeout",
    "ConnectionError": "network", "ConnectionRefusedError": "network", "ConnectionResetError": "network",
    "URLError": "network", "gaierror": "network", "NewConnectionError": "network",
    "ImportError": "import", "ModuleNotFoundError": "import",
    "HTTPError": "http", "HTTPStatusError": "http",
}


def _error_class(err_type: str) -> str:
    """Класс сбоя = ТИП исключения (структурно), с объединением родственных в семейство. По типу,
    а не по подстрокам текста — `type(e).__name__` доступен в точке записи (agent.py)."""
    t = (err_type or "").strip()
    return _TYPE_FAMILY.get(t, t or "other")


def record(name: str, ok: bool, err: str = "", err_type: str = "", args: dict | None = None) -> None:
    """Зафиксировать вызов навыка. ok=False → инкремент сбоев-подряд (по ТИПУ исключения err_type) +
    сохранить текст ошибки/аргументы (для регрессии починки). ok=True → сброс серии (навык снова жив).
    err_type — `type(e).__name__` (структурно); без него серия НЕ копится (не классифицируем вслепую)."""
    if not name:
        return
    with _LOCK:
        data = _load()
        h = data.get(name) or {"calls": 0, "failures": 0, "streak": 0,
                               "last_error": "", "last_class": "", "last_fail_args": None,
                               "status": "ok", "repairs": 0}
        h["calls"] = h.get("calls", 0) + 1
        if ok:
            h["streak"] = 0
            if h.get("status") == "degraded":
                h["status"] = "ok"  # снова заработал сам (внешний сервис ожил)
        else:
            h["failures"] = h.get("failures", 0) + 1
            # класс сбоя — ТОЛЬКО по структурному типу исключения (никакого разбора текста). Нет типа
            # → не классифицируем и серию НЕ копим (не деградируем вслепую на разнородных сбоях).
            cls = _error_class(err_type) if err_type else None
            h["streak"] = (h.get("streak", 0) + 1) if (cls and cls == h.get("last_class")) else 1
            h["last_error"] = (err or "")[:300]
            h["last_class"] = cls or ""
            if args is not None:
                h["last_fail_args"] = args
            if cls and h["streak"] >= DEGRADE_AFTER:
                h["status"] = "degraded"
        data[name] = h
        _save(data)


def health(name: str) -> dict:
    with _LOCK:
        return dict(_load().get(name) or {})


def all_health() -> dict:
    with _LOCK:
        return _load()


def degraded() -> list[str]:
    """Навыки, помеченные degraded (серия сбоев одного класса) — кандидаты на починку."""
    with _LOCK:
        return [n for n, h in _load().items() if h.get("status") == "degraded"]


def mark_repaired(name: str, success: bool) -> None:
    """После попытки починки: success → статус ok + сброс серии; иначе инкремент repairs (для cap)."""
    with _LOCK:
        data = _load()
        h = data.get(name)
        if not h:
            return
        h["repairs"] = h.get("repairs", 0) + 1
        if success:
            h["status"] = "ok"
            h["streak"] = 0
        data[name] = h
        _save(data)


def reset(name: str | None = None) -> None:
    """Сброс здоровья (тесты/ручное). name=None → всё."""
    with _LOCK:
        if name is None:
            _save({})
        else:
            data = _load()
            data.pop(name, None)
            _save(data)
