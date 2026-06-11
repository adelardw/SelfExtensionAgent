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
        ok, out = run_python_sandboxed(code, timeout=12)
        return out if ok else f"[ошибка исполнения] {out}"

    return StructuredTool.from_function(
        func=_run, name="python_exec", args_schema=_Code,
        description="Execute Python for EXACT computation (statistics, aggregation, filtering, "
                    "arithmetic with big numbers, parsing) over facts you've gathered. LLM math is "
                    "unreliable — use this for any non-trivial calculation. print() your result. "
                    "Sandboxed: stdlib only, no network/filesystem.",
    )
