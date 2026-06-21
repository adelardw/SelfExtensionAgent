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

from src.utils import run_python_sandboxed


class _Code(BaseModel):
    code: str = Field(description="Python-код для ВЫЧИСЛЕНИЙ. Выводи результат через print(). "
                                  "Доступна стандартная библиотека (math, statistics, json, re, "
                                  "datetime, itertools…). Без сети/файлов вне песочницы.")


def make_compute_tool() -> StructuredTool:
    async def _run(code: str) -> str:
        import asyncio

        from src.runtime import hitl, run_context
        # ГЕЙТ: если в прогон попал НЕДОВЕРЕННЫЙ внешний контент (веб/документ/чужой репо/MCP), а
        # песочница на macOS по умолчанию rlimits-only — инжектнутый из контента python_exec мог бы
        # читать ФС и слать наружу. Требуем HITL-подтверждение. Полный auto (явный opt-in) — пропускает.
        if run_context.external_content_seen() and not hitl.full_auto():
            try:
                approved, _note, kind = await hitl.confirm_rich(
                    "python_exec ПОСЛЕ недоверенного внешнего контента (веб/док/репо) в этом прогоне — "
                    f"возможна инъекция. Код:\n{code[:300]}")
            except Exception:  # noqa: BLE001
                approved, kind = True, "deny"  # сбой канала → не ломаем (защищает песочница)
            # БЛОКИРУЕМ только когда ЧЕЛОВЕК реально отказал (kind != 'deny'). 'deny' = нет канала
            # (headless: one-shot/eval/сервер-без-HITL) → НЕ ломаем функциональность: эксфильтрацию
            # уже режет sandbox-exec (deny network/ФС-запись). Иначе GAIA/`sea "task"` падали бы.
            if not approved and kind != "deny":
                return (f"{hitl.REFUSAL_MARK}: python_exec не подтверждён (в прогоне был недоверенный "
                        "внешний контент). Не повторяй — заверши и сообщи пользователю.")
        # no_net=True по контракту тула («без сети/ФС»). run_python_sandboxed — блокирующий
        # подпроцесс → в поток, чтобы не вешать event-loop.
        ok, out = await asyncio.to_thread(run_python_sandboxed, code, 12, True)
        return out if ok else f"[ошибка исполнения] {out}"

    return StructuredTool.from_function(
        coroutine=_run, name="python_exec", args_schema=_Code,
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
