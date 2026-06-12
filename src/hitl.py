"""
Human-in-the-loop: подтверждение side-effect действий перед исполнением — СЕМАНТИЧЕСКОЕ
и без спама.

Уровни доверия (от меньшего трения к большему):
  1. read-only тулы (config `skills.readonly`: посмотреть экран, уведомление) — без вопросов;
  2. грант («да, всегда» / config cli.allow / /auto) — однажды разрешённый skill.tool
     больше не спрашивается (персист в config.local.yml);
  3. остальное — вопрос человеку. Ответ разбирает МОДЕЛЬ (semantics.parse_reply):
     «да» → исполнить; «да, всегда» → исполнить + грант; «да, но …» → условие агенту
     (скорректировать и переспросить); «нет, …» → отказ с причиной; свободное указание
     («лучше открой gmail», «ты зацикливаешься») → следовать ему вместо плана.

Confirmer фронтенда возвращает bool (кнопки бота) или сырой текст (REPL). Нет confirmer'а
(headless) — deny by default. AGENT_DRY_RUN — независимый второй предохранитель.
"""
from __future__ import annotations

import inspect
from typing import Awaitable, Callable, Optional, Union

from omegaconf import OmegaConf

# browser_media — управление воспроизведением ПО ПРОСЬБЕ юзера («поставь на паузу»):
# подтверждать просьбу о паузе вопросом «разрешить паузу?» — абсурд.
_DEFAULT_READONLY = {"capture_screen", "analyze_screen", "notify",
                     "browser_see", "browser_read", "browser_media"}


def _load_cfg() -> tuple[bool, set[str], set[str]]:
    try:
        cfg = OmegaConf.load("config.yml")
        require = bool(cfg.get("agent", {}).get("require_confirmation", False))
        skills = set(cfg.get("skills", {}).get("confirm", []) or [])
        readonly = set(cfg.get("skills", {}).get("readonly", []) or []) or set(_DEFAULT_READONLY)
        return require, skills, readonly
    except Exception:  # noqa: BLE001
        return False, set(), set(_DEFAULT_READONLY)


REQUIRE_CONFIRMATION, CONFIRM_SKILLS, READONLY_TOOLS = _load_cfg()

# Гранты: разрешённые без вопроса skill.tool (из config cli.allow и «да, всегда» в сессии).
_grants: set[str] = set()

# Режим работы агента (три состояния, выбор юзера):
#   manual      — подтверждения и уточнения задаются человеку;
#   auto-accept — действия подтверждаются АВТОМАТИЧЕСКИ, остальное (уточнения, выбор
#                 мышления) как обычно;
#   auto        — агент автономен ЦЕЛИКОМ: сам выбирает тип мышления, сам решает
#                 развилки (допущения), действия без подтверждений.
WORK_MODES = ("manual", "auto-accept", "auto")
_work_mode: str = "manual"


def set_work_mode(mode: str) -> str:
    """Установить режим работы; неизвестное значение → manual. Возвращает применённый."""
    global _work_mode
    _work_mode = mode if mode in WORK_MODES else "manual"
    return _work_mode


def work_mode() -> str:
    return _work_mode


def is_auto() -> bool:
    """Подтверждения автоматом? (auto-accept и полный auto)."""
    return _work_mode in ("auto-accept", "auto")


def full_auto() -> bool:
    """Полная автономия: агент сам выбирает мышление и решает развилки."""
    return _work_mode == "auto"


def set_auto(value: bool) -> None:
    """Совместимость: True → auto-accept, False → manual."""
    set_work_mode("auto-accept" if value else "manual")


def load_grants(keys) -> None:
    """Гранты из конфига (cli.allow) при старте фронтенда."""
    _grants.update(str(k) for k in (keys or []))


def grant(key: str, persist: bool = True) -> None:
    """Разрешить skill.tool без дальнейших вопросов; persist → в config.local.yml."""
    _grants.add(key)
    if persist:
        try:
            from .cli_config import get_cli, set_cli
            allow = list(get_cli("allow") or [])
            if key not in allow:
                allow.append(key)
                set_cli("allow", allow)
        except Exception:  # noqa: BLE001
            pass


Confirmer = Callable[[str], Union[bool, str, Awaitable]]
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


async def confirm_rich(description: str) -> tuple[bool, str, str]:
    """
    Спросить человека. Возвращает (approved, note, kind), kind:
      'yes' / 'always' — разрешено (always — ещё и грант у вызывающего);
      'condition' — согласие с условием: НЕ исполнять как есть, условие в note;
      'no' — отказ (note — причина); 'redirect' — указание вместо да/нет (note — что делать);
      'deny' — канала нет (headless): deny by default, НЕ выбор юзера (не учимся).
    """
    if _confirmer is None:
        return False, "", "deny"
    try:
        res = _confirmer(description)
        if inspect.isawaitable(res):
            res = await res
    except Exception:  # noqa: BLE001
        res = False
    if isinstance(res, bool):
        kind, note = ("yes" if res else "no"), ""
    else:
        from .semantics import parse_reply
        kind, note = await parse_reply(str(res), action=description)
    approved = kind in ("yes", "always")
    # Решение человека — сигнал контура (implicit feedback): в журнал прогона → эпизод →
    # отказы становятся фактами профиля. deny-by-default не пишем — это не выбор юзера.
    try:
        from . import interaction
        interaction.record_hitl(description, approved, note=note)
    except Exception:  # noqa: BLE001
        pass
    return approved, note, kind


async def confirm(description: str) -> bool:
    """True — человек разрешил. Нет confirmer'а → deny by default. (Совместимость.)"""
    approved, _note, _kind = await confirm_rich(description)
    return approved


def wrap_with_confirmation(t, skill_name: str):
    """Оборачивает LangChain-tool уровнями доверия: read-only и грантованное идёт сразу,
    остальное — семантическое подтверждение (см. докстринг модуля)."""
    from langchain_core.tools import StructuredTool

    async def _arun(**kwargs):
        key = f"{skill_name}.{t.name}"
        if t.name in READONLY_TOOLS or is_auto() or key in _grants:
            return await t.ainvoke(kwargs)  # доверено: без вопроса
        args_short = ", ".join(f"{k}={str(v)[:80]}" for k, v in kwargs.items())
        approved, note, kind = await confirm_rich(f"{key}({args_short})")
        if approved:
            if kind == "always":
                grant(key)  # «да, всегда» → этот тул больше не спрашиваем (персист)
            return await t.ainvoke(kwargs)
        if kind == "condition":
            return (f"Пользователь готов разрешить ТОЛЬКО при условии: «{note}». "
                    f"Скорректируй вызов {t.name} под это условие и вызови снова "
                    "(будет новое подтверждение). Если условие невыполнимо — сообщи об этом.")
        if kind == "redirect":
            return (f"Пользователь вместо подтверждения дал указание: «{note}». "
                    f"Вызов {t.name} НЕ выполнен. Следуй указанию пользователя — оно "
                    "приоритетнее изначального плана.")
        reason = f" Причина: «{note}»." if note else ""
        return (
            f"{REFUSAL_MARK}: пользователь не подтвердил вызов {t.name}.{reason} "
            "НЕ повторяй этот и любые похожие вызовы — сразу заверши и сообщи пользователю, "
            "что действие требует его подтверждения."
        )

    return StructuredTool(
        name=t.name,
        description=t.description,
        args_schema=t.args_schema,
        coroutine=_arun,
    )
