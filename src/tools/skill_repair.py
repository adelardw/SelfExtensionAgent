"""
Самопочинка навыков (замыкание контура здоровья).

Когда навык помечен `degraded` (серия сбоев одного класса — напр. внешний API сменил
endpoint/схему), здесь его пробуем ПОЧИНИТЬ: умная модель переписывает код навыка по тексту
ошибки → СУЩЕСТВУЮЩИЕ гейты (AST `_validate_python` + `_security_gate`) → smoke-подпроцесс с
ПОСЛЕДНИМ ПАДАВШИМ вызовом (run_tool_sandboxed, сеть включена) → применяем ТОЛЬКО если smoke
реально прошёл. Иначе откат: если сервис просто лёг, никакой переписью не поможешь, и health
сам вернёт `ok` при следующем успехе.

Не новый риск: переиспользует smoke-sandbox / AST-гейт / registry, что и ручное создание навыка.
Триггерится в ФОНЕ из сервера после прогона (не в графе → нет user-latency). Cap MAX_REPAIRS от
бесконечного галлюцинаторного переписывания.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from src.tools import skill_health

MAX_REPAIRS = 3  # потолок попыток починки навыка (анти-бесконечная переписка)


def _owning_skill(tool_name: str) -> str | None:
    """Навык-владелец тула (имя тула ≠ имя навыка в общем случае). Редкая операция → перебор реестра."""
    from src.tools.skill_creation import _merged_registry, get_all_loaded_skill_tools
    for sk in _merged_registry():
        try:
            if any(getattr(t, "name", None) == tool_name for t in get_all_loaded_skill_tools([sk])):
                return sk
        except Exception:  # noqa: BLE001
            continue
    return None


_REPAIR_PROMPT = (
    "Ты чинишь СЛОМАВШИЙСЯ Python-навык агента. Навык падает в проде — скорее всего внешний "
    "API/endpoint/схема ответа изменились, либо парсинг устарел.\n\n"
    "Имя тула: {tool}\nКласс ошибки: {cls}\nТекст ошибки: {err}\n"
    "Аргументы падавшего вызова: {args}\n\n"
    "ТЕКУЩИЙ КОД:\n```python\n{code}\n```\n\n"
    "Верни ИСПРАВЛЕННЫЙ полный код навыка. Требования: СОХРАНИ то же имя @tool-функции и сигнатуру; "
    "почини причину сбоя (новый endpoint/поле/формат/таймаут-ретрай); только стандартные библиотеки "
    "(urllib, json, re) или то, что уже было; без сети-эксфильтрации. Верни ТОЛЬКО Python-код, без "
    "пояснений и markdown-ограждений."
)


def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rsplit("```", 1)[0]
    return t.strip()


async def repair_tool(tool_name: str) -> dict:
    """Починить навык, владеющий tool_name. Возвращает {ok, skill, reason}."""
    from src.llm.llm import chat
    from src.tools.skill_creation import update_skill_tools, _validate_python, _security_gate, _skill_base

    h = skill_health.health(tool_name)
    if h.get("repairs", 0) >= MAX_REPAIRS:
        return {"ok": False, "skill": None, "reason": f"cap: уже {h['repairs']} попыток починки"}
    skill = _owning_skill(tool_name)
    if not skill:
        return {"ok": False, "skill": None, "reason": "навык-владелец не найден в реестре"}

    code_file = _skill_base(skill) / f"{skill}.py"
    if not code_file.exists():
        return {"ok": False, "skill": skill, "reason": "файл кода навыка не найден"}
    cur_code = code_file.read_text("utf-8")
    args = h.get("last_fail_args") or {}

    # 1) умная модель переписывает
    try:
        resp = await chat("deep", 0).ainvoke(_REPAIR_PROMPT.format(
            tool=tool_name, cls=h.get("last_class", "?"), err=h.get("last_error", "?"),
            args=json.dumps(args, ensure_ascii=False), code=cur_code[:6000]))
        new_code = _strip_fence(resp.content if hasattr(resp, "content") else str(resp))
    except Exception as e:  # noqa: BLE001
        skill_health.mark_repaired(tool_name, success=False)
        return {"ok": False, "skill": skill, "reason": f"LLM-переписывание упало: {type(e).__name__}"}

    if not new_code or new_code == cur_code:
        skill_health.mark_repaired(tool_name, success=False)
        return {"ok": False, "skill": skill, "reason": "пустой/идентичный код"}

    # 2) ГЕЙТЫ (как при ручном создании): AST + security ДО любого применения
    ok_ast, ast_err = _validate_python(new_code)
    if not ok_ast:
        skill_health.mark_repaired(tool_name, success=False)
        return {"ok": False, "skill": skill, "reason": f"AST-гейт отклонил: {ast_err[:120]}"}
    safe, sec = _security_gate(new_code)
    if not safe:
        skill_health.mark_repaired(tool_name, success=False)
        return {"ok": False, "skill": skill, "reason": f"security-гейт отклонил: {sec[:120]}"}

    # 3) РЕГРЕССИЯ: smoke с ПОСЛЕДНИМ ПАДАВШИМ вызовом во ВРЕМЕННОМ файле (не трогаем боевой, пока не
    # убедились). Сеть включена — навык легитимно ходит в свой API. Прошло → причина была в коде.
    from src.utils import run_tool_sandboxed
    tmp = Path(tempfile.mkdtemp()) / f"{skill}.py"
    tmp.write_text(new_code, "utf-8")
    try:
        smoke_ok, smoke_out = await asyncio.to_thread(
            run_tool_sandboxed, tmp, tool_name, args, 15, False)
    except Exception as e:  # noqa: BLE001
        smoke_ok, smoke_out = False, f"{type(e).__name__}: {e}"

    if not smoke_ok:
        # не применяем: либо рерайт не помог, либо сервис всё ещё лёг (тогда health сам вернёт ok)
        skill_health.mark_repaired(tool_name, success=False)
        return {"ok": False, "skill": skill, "reason": f"smoke не прошёл: {str(smoke_out)[:120]}"}

    # 4) применяем боевым путём (тот же гейт ещё раз внутри) + помечаем здоровье
    msg = update_skill_tools(skill, new_code)
    skill_health.mark_repaired(tool_name, success=True)
    return {"ok": True, "skill": skill, "reason": f"починен и прошёл smoke ({str(msg)[:60]})"}


async def heal_degraded() -> list[dict]:
    """Починить ВСЕ деградировавшие навыки (фон после прогона / maintenance). Бережно, по одному."""
    results = []
    for tool_name in skill_health.degraded():
        try:
            results.append(await repair_tool(tool_name))
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "skill": tool_name, "reason": f"{type(e).__name__}: {e}"})
    return results


def heal_degraded_sync() -> list[dict]:
    """Синхронная обёртка для maintenance-CLI."""
    return asyncio.run(heal_degraded())
