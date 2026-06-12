"""
Human-in-the-loop: подтверждение side-effect действий перед исполнением.

Тулы навыков из списка config `skills.confirm` (device/app/phone-класс) оборачиваются
так, что перед реальным вызовом запрашивается подтверждение у человека через
зарегистрированный confirmer (REPL — input, Telegram — inline-кнопки). Если
confirmer не зарегистрирован (например, headless-сервер) — действие ОТКЛОНЯЕТСЯ
(deny by default): автономный процесс не должен молча трогать устройство.

Включается config `agent.require_confirmation: true`. AGENT_DRY_RUN остаётся
независимым вторым предохранителем внутри самих навыков.
"""
from __future__ import annotations

import inspect
from typing import Awaitable, Callable, Optional, Union

from omegaconf import OmegaConf


def _load_cfg() -> tuple[bool, set[str]]:
    try:
        cfg = OmegaConf.load("config.yml")
        require = bool(cfg.get("agent", {}).get("require_confirmation", False))
        skills = set(cfg.get("skills", {}).get("confirm", []) or [])
        return require, skills
    except Exception:  # noqa: BLE001
        return False, set()


REQUIRE_CONFIRMATION, CONFIRM_SKILLS = _load_cfg()

Confirmer = Callable[[str], Union[bool, Awaitable[bool]]]
_confirmer: Optional[Confirmer] = None


def set_confirmer(fn: Optional[Confirmer]) -> None:
    """Регистрирует канал подтверждения текущего фронтенда (REPL/бот)."""
    global _confirmer
    _confirmer = fn


# Маркер отклонённого пользователем действия — по нему граф понимает, что это НЕ
# провал агента, а сознательный отказ: не ретраить и не винить ноды (см. step_executor).
REFUSAL_MARK = "⛔ОТКЛОНЕНО"


def needs_confirmation(skill_name: str) -> bool:
    """Side-effect навыки из config + ЛЮБОЙ импортированный (сторонний) скилл."""
    if not REQUIRE_CONFIRMATION:
        return False
    if skill_name in CONFIRM_SKILLS:
        return True
    try:  # импортированные OpenClaw-скиллы — сторонний код/CLI → всегда под подтверждением
        from .tools.skill_creation import _load_registry

        return bool(_load_registry().get(skill_name, {}).get("imported"))
    except Exception:  # noqa: BLE001
        return False


async def confirm(description: str) -> bool:
    """True — человек разрешил. Нет confirmer'а → deny by default."""
    approved = False
    if _confirmer is not None:
        try:
            res = _confirmer(description)
            if inspect.isawaitable(res):
                res = await res
            approved = bool(res)
        except Exception:  # noqa: BLE001
            approved = False
    # Решение человека — сигнал контура (implicit feedback): копится в журнале прогона,
    # reflect пишет его в эпизод, отказы становятся фактами профиля. Кроме deny-by-default
    # без канала (headless): это не выбор юзера — не учиться на нём.
    if _confirmer is not None:
        try:
            from . import interaction
            interaction.record_hitl(description, approved)
        except Exception:  # noqa: BLE001
            pass
    return approved


def wrap_with_confirmation(t, skill_name: str):
    """Оборачивает LangChain-tool: вызов идёт только после подтверждения человеком."""
    from langchain_core.tools import StructuredTool

    async def _arun(**kwargs):
        args_short = ", ".join(f"{k}={str(v)[:80]}" for k, v in kwargs.items())
        if not await confirm(f"{skill_name}.{t.name}({args_short})"):
            return (
                f"{REFUSAL_MARK}: пользователь не подтвердил вызов {t.name}. "
                "НЕ повторяй этот и любые похожие вызовы — сразу заверши и сообщи пользователю, "
                "что действие требует его подтверждения."
            )
        return await t.ainvoke(kwargs)

    return StructuredTool(
        name=t.name,
        description=t.description,
        args_schema=t.args_schema,
        coroutine=_arun,
    )
