"""
Журнал взаимодействий (interaction ledger) — стадия «↓ сигнал» замкнутого контура.

События взаимодействия УЖЕ происходят (HITL-подтверждения/отказы в hitl.py, ответы на
уточнения в clarify.py), но раньше умирали в конце прогона. Здесь они копятся за прогон
(contextvar, изолирован между запросами — как clarify-ledger), reflect пишет их в эпизод
(сырьё для per-user backward и бандитов), а harvest БЕЗ LLM конвертирует в персонализацию:

  HITL-отказ      → факт профиля («не делать X без явной просьбы») — агент перестаёт
                    предлагать то, что юзер отвергает: implicit feedback из отказа.
  clarify-ответ   → факт профиля (вопрос→ответ) — онбординг-по-исполнению становится
                    НАКОПИТЕЛЬНЫМ: однажды уточнённое не переспрашивается в новых сессиях.

Ноль LLM-вызовов в горячем пути: запись — append в список, harvest — в reflect.
"""
from __future__ import annotations

import contextvars
import time

_ledger: contextvars.ContextVar[list | None] = contextvars.ContextVar("interaction_ledger", default=None)


def reset_ledger() -> None:
    """Новый прогон — чистый журнал (зовётся в recall_node, рядом с clarify.reset_ledger)."""
    _ledger.set([])


def _cur() -> list:
    cur = _ledger.get()
    if cur is None:
        cur = []
        _ledger.set(cur)
    return cur


def record_hitl(action: str, approved: bool) -> None:
    """HITL-решение по side-effect действию. action = 'skill.tool(args…)'."""
    _cur().append({"type": "hitl", "action": action, "approved": approved, "ts": time.time()})


def events() -> list[dict]:
    return list(_cur())


def _tool_of(action: str) -> str:
    """'skill.tool(args…)' → 'skill.tool' (ключ факта: упорство по ИНСТРУМЕНТУ, не аргументам)."""
    return action.split("(", 1)[0].strip() or action[:60]


def harvest(store, user_id: str, clarify_items: list[dict] | None = None) -> int:
    """
    События прогона → персонализация (факты профиля). Возвращает число записанных фактов.
    Без LLM: правила прямые, обратимые (факты видны в /facts, upsert по ключу — без дублей).
    """
    n = 0
    for ev in _cur():
        if ev["type"] == "hitl" and not ev["approved"]:
            tool = _tool_of(ev["action"])
            store.add_fact(
                user_id=user_id,
                key=f"hitl-отказ: {tool}",
                value=(f"Пользователь отклонил действие {ev['action'][:160]} — не выполнять и "
                       "не предлагать подобное без его явной просьбы; сначала спрашивать."),
                importance=0.7,
                tags=["hitl", "preference"],
            )
            n += 1
    # Ответы на уточнения (только answered — допущения агента не факты о юзере).
    for it in (clarify_items or []):
        if it.get("status") == "answered" and it.get("question") and it.get("answer"):
            store.add_fact(
                user_id=user_id,
                key=f"уточнение: {it['question'][:80]}",
                value=str(it["answer"])[:300],
                importance=0.55,
                tags=["clarify", "onboarding"],
            )
            n += 1
    return n
