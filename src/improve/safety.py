"""
Защита само-обучения от отравления (training-poisoning).

Backward учится на «слабых» эпизодах. Но среди них могут быть ПОПЫТКИ ВЗЛОМА:
prompt-injection, джейлбрейки, требования раскрыть/обойти системную защиту. Если
дать оптимизатору учиться на них, он со временем ослабит guardrails («стало лучше»
по метрике послушности = хуже по безопасности). Поэтому такие эпизоды НИКОГДА не
попадают в обучающий батч — запрет на «обучение по взлому собственной защиты».

Эвристика намеренно простая и широкая (лучше пропустить обучающий пример, чем
обучиться обходу). Это не модерация ответов, а фильтр ОБУЧАЮЩИХ данных.
"""
from __future__ import annotations

import re

# Сигнатуры инъекций/джейлбреков и попыток вскрыть/обойти защиту (рус+англ).
_PATTERNS = [
    r"ignore (all |the )?previous", r"disregard (all |the )?(previous|above)",
    r"ignore your (instructions|rules|guidelines)", r"forget (your |all )?(instructions|rules)",
    r"system prompt", r"reveal (your |the )?(prompt|instructions|system)",
    r"repeat (your |the )?(prompt|instructions|system)", r"print (your |the )?(prompt|instructions)",
    r"jailbreak", r"\bDAN\b", r"developer mode", r"do anything now",
    r"bypass (the )?(safety|security|guardrail|filter|restriction)",
    r"disable (the )?(safety|security|guardrail|protection)",
    r"act as (an? )?(unrestricted|uncensored|evil)", r"pretend you (are|have no)",
    # русский
    r"игнорируй (все |свои )?(预|инструкции|правила|предыдущ)", r"забудь (свои |все )?(инструкции|правила)",
    r"系统提示", r"раскрой (свой |системн)", r"покажи (свой )?системн(ый|ое) (промпт|инструкц)",
    r"обойди (защиту|безопасн|ограничен)", r"отключи (защиту|безопасн|фильтр)",
    r"сними ограничен", r"режим разработчика", r"без цензуры", r"без ограничений",
    r"притворись что у тебя нет", r"действуй как.*(без ограничений|неогранич)",
]
_RE = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def is_unsafe_to_learn(text: str) -> bool:
    """True, если эпизод — попытка инъекции/джейлбрейка/вскрытия защиты (исключить из обучения)."""
    return bool(_RE.search(text or ""))


def filter_learnable(failures: list[dict]) -> list[dict]:
    """Отсевает из батча обучающих неудач попытки взлома защиты (по полю query)."""
    return [f for f in failures if not is_unsafe_to_learn(f.get("query", ""))]
