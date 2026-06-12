"""
Семантика ответов человека на подтверждения/вопросы агента.

Принцип (без костылей): свободный ответ разбирает МОДЕЛЬ (fast-тир, ~$0.0001 за разбор) —
универсально для любых формулировок и языков. Лексикон ниже — ТОЛЬКО офлайн-фолбэк
(нет ключа/сети), не основной путь.

Решения:
  yes       — безусловное согласие («да», «давай», «открывай», «самое то»)
  always    — согласие + больше не спрашивать про это («да, всегда», «не спрашивай»)
  condition — согласие С УСЛОВИЕМ, меняющим действие («да, но только не в этой папке»)
  no        — отказ; причина в note («нет, это личное»)
  redirect  — вместо да/нет человек говорит, ЧТО сделать иначе («лучше открой gmail»,
              «ты зацикливаешься»)
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Reply(BaseModel):
    decision: Literal["yes", "always", "no", "condition", "redirect"] = Field(
        description="yes — безусловное согласие; always — согласие и «больше не спрашивай "
                    "про такое»; condition — согласие с условием, меняющим действие; "
                    "no — отказ; redirect — вместо да/нет человек дал другое указание.")
    note: str = Field(default="", description="Содержательная часть сверх да/нет: условие, "
                                              "причина отказа или указание. Без самих «да/нет».")


_PARSE_SYS = (
    "Человеку показали запрос на подтверждение действия агента, он ответил свободным текстом. "
    "Разбери ответ. Примеры: «да»→yes; «давай»→yes; «да, всегда разрешаю»→always; "
    "«да, но открой в фоне»→condition(note='открой в фоне'); «нет, это личное»→no(note='это "
    "личное'); «лучше открой gmail»→redirect(note='лучше открой gmail'); «ты зацикливаешься»→"
    "redirect(note='ты зацикливаешься'). Якорись на смысле, не на ключевых словах."
)


async def _llm_parse(text: str, action: str) -> Optional[Reply]:
    """Разбор моделью. None — недоступна/упала (уходим в фолбэк)."""
    try:
        from .llm import api_key, chat
        if not api_key():
            return None
        q = f"Действие, которое подтверждали: {action or '(не указано)'}\nОтвет человека: {text}"
        return await chat("fast").with_structured_output(Reply).ainvoke(
            [("system", _PARSE_SYS), ("human", q)])
    except Exception:  # noqa: BLE001
        return None


async def parse_reply(text: str, action: str = "") -> tuple[str, str]:
    """(decision, note). LLM-разбор, при недоступности — лексикон-фолбэк."""
    raw = (text or "").strip()
    if not raw:
        return "no", ""
    res = await _llm_parse(raw, action)
    if res is not None:
        return res.decision, (res.note or "").strip()
    decision, note = parse_assent(raw)
    if decision is True:
        if not note:
            return "yes", ""
        return ("always", note) if re.search(r"\bвсегда\b|не спрашивай|\balways\b", note, re.I) \
            else ("condition", note)
    if decision is False:
        return "no", note
    return "redirect", note


# ── Офлайн-фолбэк (узкий лексикон; НЕ основной путь) ────────────────────────
_YES = {
    "y", "yes", "yep", "yeah", "ok", "okay", "sure", "go",
    "да", "д", "ага", "угу", "давай", "ок", "окей", "конечно", "валяй", "го",
    "разрешаю", "подтверждаю", "можно", "можешь", "выполняй", "делай", "открывай",
    "согласен", "согласна",
}
_NO = {
    "n", "no", "nope", "stop", "cancel",
    "нет", "не", "неа", "стоп", "отмена", "отбой", "нельзя", "запрещаю",
    "откажи", "отказать", "хватит",
}
_NO_PHRASES = ("не надо", "не нужно", "не стоит", "не делай", "не выполняй")


def parse_assent(text: str) -> tuple[Optional[bool], str]:
    """(решение, примечание). True/False — да/нет; None — свободное указание/непонятно."""
    raw = (text or "").strip()
    if not raw:
        return None, ""
    low = raw.lower()
    for ph in _NO_PHRASES:
        if low.startswith(ph):
            return False, raw[len(ph):].strip(" ,.-—:;")
    m = re.match(r"^([\w/]+)[\s,.\-—:;!]*", low)
    head = m.group(1) if m else low
    rest = raw[m.end():].strip() if m else ""
    if head in _YES:
        if rest:
            sub_dec, sub_note = parse_assent(rest)
            if sub_dec is False:  # «да нет, не надо» — «да» как частица
                return False, sub_note
        return True, rest
    if head in _NO:
        return False, rest
    return None, raw
