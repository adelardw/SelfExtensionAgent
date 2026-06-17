"""
Вычислительный слой агента: исполнение Python в песочнице как ИНСТРУМЕНТ.

Закрывает «вычислительный» пробел held-out: задачи требуют не только найти факты, но и
ПОСЧИТАТЬ над ними (статистика, агрегация, фильтры, арифметика с большими числами, парсинг).
LLM арифметику/агрегацию делает ненадёжно — код делает точно. Код идёт в изолированный
подпроцесс (rlimits + опц. syscall-изоляция + wall-kill), НЕ в процессе агента.
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .utils import run_python_sandboxed


class _Code(BaseModel):
    code: str = Field(description="Python-код для ВЫЧИСЛЕНИЙ. Выводи результат через print(). "
                                  "Доступна стандартная библиотека (math, statistics, json, re, "
                                  "datetime, itertools…). Без сети/файлов вне песочницы.")


def make_compute_tool() -> StructuredTool:
    def _run(code: str) -> str:
        # no_net=True по контракту тула («без сети/ФС») + B2: python_exec всегда доступен, без HITL,
        # код от LLM/инъекции → самый широкий канал эксфильтрации. Сеть режем (где есть syscall-sandbox).
        ok, out = run_python_sandboxed(code, timeout=12, no_net=True)
        return out if ok else f"[ошибка исполнения] {out}"

    return StructuredTool.from_function(
        func=_run, name="python_exec", args_schema=_Code,
        description="Execute Python for EXACT computation (statistics, aggregation, filtering, "
                    "arithmetic with big numbers, parsing) over facts you've gathered. LLM math is "
                    "unreliable — use this for any non-trivial calculation. print() your result. "
                    "Sandboxed: stdlib only, no network/filesystem.",
    )


class _TZ(BaseModel):
    timezone: str = Field(default="UTC", description="IANA timezone name, e.g. 'Europe/Moscow', "
                          "'UTC', 'America/New_York', 'Asia/Tokyo'. City→zone: Moscow=Europe/Moscow, "
                          "London=Europe/London, New York=America/New_York.")


def make_datetime_tool() -> StructuredTool:
    """Текущие дата/время из системных часов в нужном часовом поясе (IANA).

    Закрывает тривиальный, но частый пробел: на вопрос «сколько сейчас времени / какое число»
    модель без часов уходит в веб и советует сайты-часы (баг: 120с в act, ноль пользы). Тул даёт
    точный ответ мгновенно и детерминированно (system clock + zoneinfo)."""
    def _now(timezone: str = "UTC") -> str:
        from datetime import datetime
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(timezone)
        except Exception:  # noqa: BLE001 — неизвестная зона/нет tzdata
            return (f"Unknown timezone '{timezone}'. Use an IANA name like 'Europe/Moscow', 'UTC', "
                    f"'America/New_York'.")
        now = datetime.now(tz)
        return (now.strftime("%Y-%m-%d %H:%M:%S") + f" {timezone} (UTC{now.strftime('%z')}), "
                + now.strftime("%A"))

    return StructuredTool.from_function(
        func=_now, name="current_datetime", args_schema=_TZ,
        description="Get the CURRENT real date and time from the system clock for a given IANA "
                    "timezone (e.g. Europe/Moscow, UTC, America/New_York). Use this WHENEVER asked "
                    "'what time/date is it' anywhere — answer directly, do NOT recommend websites or "
                    "guess. Moscow = Europe/Moscow (UTC+3).",
    )
