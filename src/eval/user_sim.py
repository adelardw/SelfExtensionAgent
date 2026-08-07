"""Мульти-агентная симуляция пользователей: LLM-персоны с уникальным профилем/стилем ведут
МНОГОХОДОВЫЕ диалоги с реальным графом агента, реагируя на его настоящие ответы (уникальный
UX-фидбек на каждом ходе), затем судья оценивает весь диалог против цели персоны.

Архитектура — оркестрация ЧИСТАЯ (run_dialogue/aggregate тестируются без LLM: все внешние
вызовы инжектятся корутинами), LLM-обвязка и env-изоляция живут в раннере bench_sim_users.py.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Optional

from pydantic import BaseModel, Field

# ── персоны: уникальный профиль + стиль + сценарий (цель и первая реплика) ──────
# Каждая проверяет СВОЙ контур агента: глубина+источники (research), короткий act,
# нечёткость (clarify), данные+файл (fetch_data/export). patience = потолок ходов.
PERSONAS: list[dict] = [
    {
        "id": "lena_researcher",
        "name": "Лена, аспирантка-исследователь",
        "profile": "Пишет диссертацию, ценит ГЛУБИНУ и ссылки на источники. Раздражается на "
                   "воду и ответы без ссылок. Задаёт уточняющие вопросы вглубь по содержанию.",
        "style": "вежливая, развёрнутые вопросы, академическая лексика",
        "patience": 3,
        "scenario": {
            "goal": "Получить обзор свежих методов борьбы с галлюцинациями LLM со ссылками на "
                    "конкретные работы/статьи и понять, какие метрики там используют.",
            "opening": "Привет! Собираю материал для обзора: какие сейчас основные подходы к "
                       "борьбе с галлюцинациями в LLM? Желательно со ссылками на конкретные "
                       "работы или статьи.",
            "success": "названы конкретные подходы И даны ссылки на источники",
        },
    },
    {
        "id": "marat_busy",
        "name": "Марат, занятой предприниматель",
        "profile": "Время — деньги. Хочет короткий конкретный ответ с цифрами и ссылками, "
                   "СРАЗУ. Длинную воду не читает, пишет «короче». Ценит топ-N списки с ценами.",
        "style": "короткие рубленые фразы, без приветствий, может быть резким",
        "patience": 3,
        "scenario": {
            "goal": "Выбрать сервис онлайн-бухгалтерии для ИП: топ-3 с актуальными ценами и "
                    "ссылками, затем сравнение двух лидеров по цене за год.",
            "opening": "топ-3 сервиса бухгалтерии для ИП на упрощёнке, цены и ссылки. коротко.",
            "success": "получен топ с ценами и ссылками, сравнение двух вариантов",
        },
    },
    {
        "id": "tamara_novice",
        "name": "Тамара, новичок в технологиях",
        "profile": "Далека от IT, формулирует размыто, своими словами. На встречные вопросы "
                   "отвечает неточно. Хочет, чтобы её ПОНЯЛИ и объяснили простыми словами, "
                   "без жаргона.",
        "style": "разговорная, сумбурная, без терминов, многоточия",
        "patience": 3,
        "scenario": {
            "goal": "Понять простыми словами, как сделать, чтобы напоминания о днях рождения "
                    "родни появлялись сами в телефоне.",
            "opening": "слушай, мне надо чтобы телефон сам напоминал про дни рождения ну там "
                       "родственники все дела... это вообще как делается? только попроще объясни",
            "success": "получена понятная пошаговая инструкция без жаргона, агент уточнил "
                       "детали (какой телефон) вместо угадывания",
        },
    },
    {
        "id": "igor_analyst",
        "name": "Игорь, дата-аналитик",
        "profile": "Работает с данными, хочет ЧИСЛА и ФАЙЛ, а не пересказ. Проверяет, откуда "
                   "данные (источник обязателен). Не терпит выдуманных чисел.",
        "style": "деловой, точный, требует конкретику и формат",
        "patience": 3,
        "scenario": {
            "goal": "Получить курс доллара ЦБ за последний месяц: откуда данные, сами значения "
                    "и файл-таблицу (xlsx/csv).",
            "opening": "Нужен курс доллара ЦБ РФ за последний месяц по дням: источник данных, "
                       "и собери мне это в файл-таблицу.",
            "success": "указан источник, данные реальные (не выдуманные), файл предложен/собран",
        },
    },
]


class PersonaTurn(BaseModel):
    """Ход персоны: реакция на ответ агента → внутренний UX-фидбек + следующая реплика."""
    satisfaction: int = Field(description="Удовлетворённость ПОСЛЕДНИМ ответом агента, 1-5")
    feedback: str = Field(description="Короткая UX-заметка персоны об ответе: что зашло/бесит "
                                      "(её глазами, её стилем)")
    done: bool = Field(description="True — цель достигнута ИЛИ персона сдалась; диалог завершён")
    message: str = Field(description="Следующая реплика ПОЛЬЗОВАТЕЛЯ агенту в стиле персоны "
                                     "(пустая строка, если done)")


class DialogueVerdict(BaseModel):
    """Вердикт судьи по ВСЕМУ диалогу против цели персоны."""
    goal_achieved: bool = Field(description="Достигнута ли цель сценария по критерию success")
    grounded: bool = Field(description="Опирались ли ответы на источники/ссылки/реальные данные "
                                       "(не выдумка и не пересказ памяти)")
    depth: int = Field(description="Глубина и полнота ответов, 1-5")
    clarity: int = Field(description="Соответствие стилю/уровню персоны (понятность ей), 1-5")
    ux_issues: list[str] = Field(description="Конкретные UX-проблемы, замеченные в диалоге")
    highlight: str = Field(description="Что агент сделал особенно хорошо (пусто, если нечего)")


def persona_system_prompt(p: dict) -> str:
    """Системный промпт персоны-симулятора: играть ЖИВОГО пользователя, не ассистента."""
    s = p["scenario"]
    return (
        f"Ты играешь РОЛЬ пользователя AI-ассистента. Ты — {p['name']}.\n"
        f"Кто ты: {p['profile']}\n"
        f"Твой стиль письма: {p['style']}. Пиши ТОЛЬКО в этом стиле, от первого лица.\n"
        f"Твоя цель в диалоге: {s['goal']}\n"
        f"Цель достигнута, если: {s['success']}.\n\n"
        "Правила:\n"
        "• Реагируй на РЕАЛЬНОЕ содержание последнего ответа ассистента: дожимай, уточняй, "
        "возражай — как живой человек с твоим характером.\n"
        "• Если ответ плохой (вода, нет ссылок, не по делу) — скажи об этом В СВОЁМ стиле и "
        "потребуй нужное; satisfaction ставь честно низкий.\n"
        "• done=True когда цель реально достигнута ИЛИ терпение кончилось. Не тяни диалог "
        "ради диалога.\n"
        "• message — ТОЛЬКО текст реплики пользователя, без мета-комментариев."
    )


def format_transcript(history: list[dict]) -> str:
    """[{'role','content'}] → читаемый транскрипт для персоны/судьи."""
    lines = []
    for h in history:
        who = "ПОЛЬЗОВАТЕЛЬ" if h.get("role") == "user" else "АССИСТЕНТ"
        lines.append(f"{who}: {str(h.get('content', ''))[:1500]}")
    return "\n\n".join(lines) or "(диалог пуст)"


async def run_dialogue(
    agent_call: Callable[[str, list[dict], str], Awaitable[str]],
    persona_step: Callable[[dict, str], Awaitable[PersonaTurn]],
    persona: dict,
    thread_id: str,
    max_turns: Optional[int] = None,
) -> dict:
    """Один диалог: opening → agent → persona (фидбек+реплика) → agent → … до done/потолка.
    Все внешние вызовы инжектятся (тестируемо без LLM). Возвращает историю, satisfaction по
    ходам и UX-фидбек персоны."""
    turns_cap = max_turns or persona.get("patience", 3)
    history: list[dict] = []
    satisfactions: list[int] = []
    feedbacks: list[str] = []
    message = persona["scenario"]["opening"]

    for _ in range(turns_cap):
        history.append({"role": "user", "content": message})
        answer = await agent_call(message, list(history), thread_id)
        history.append({"role": "assistant", "content": answer})

        turn = await persona_step(persona, format_transcript(history))
        satisfactions.append(int(turn.satisfaction))
        if turn.feedback:
            feedbacks.append(turn.feedback)
        if turn.done or not (turn.message or "").strip():
            break
        message = turn.message

    return {
        "persona": persona["id"],
        "history": history,
        "turns": len(satisfactions),
        "satisfaction": satisfactions,
        "feedbacks": feedbacks,
    }


def aggregate(results: list[dict]) -> dict:
    """Сводка по прогонам: {n, achieved, grounded, avg_satisfaction, avg_depth, issues}."""
    n = len(results)
    if not n:
        return {"n": 0, "achieved": 0, "grounded": 0, "avg_satisfaction": 0.0,
                "avg_depth": 0.0, "issues": []}
    sats = [s for r in results for s in r.get("satisfaction", [])]
    verdicts = [r.get("verdict") for r in results if r.get("verdict")]
    issues = [i for v in verdicts for i in (v.get("ux_issues") or [])]
    return {
        "n": n,
        "achieved": sum(1 for v in verdicts if v.get("goal_achieved")),
        "grounded": sum(1 for v in verdicts if v.get("grounded")),
        "avg_satisfaction": round(sum(sats) / len(sats), 2) if sats else 0.0,
        "avg_depth": round(sum(v.get("depth", 0) for v in verdicts) / len(verdicts), 2)
        if verdicts else 0.0,
        "issues": issues,
    }
