"""
Реестр уточнений (clarification ledger) — онбординг неясных запросов как СВОЙСТВО
СИСТЕМЫ, а не отдельная нода.

Неоднозначность всплывает в разные моменты: на входе (reflexion ambiguity-гейт),
при планировании (clarify_gate — батч вопросов до старта) и прямо в исполнении
(инструмент ask_user — догон, когда шаг упёрся в развилку). Все вопросы и ответы
копятся в ОДИН ledger на прогон (contextvar, изолирован между запросами) и
инъектируются во все ноды ниже по течению — агент не переспрашивает дважды.

Формат вопроса гибкий: где набор вариантов конечен — даём маркеры (options),
где нет — открытый вопрос. Канал к человеку регистрирует фронтенд (REPL/бот).
Если канала нет или ответа нет — берём разумное допущение (assume) и помечаем
пункт как assumed, чтобы финальный ответ это отразил («исходил из того, что…»).
"""
from __future__ import annotations

import contextvars
import inspect
from typing import Awaitable, Callable, Optional, Union

# Ledger одного прогона. contextvar → разные запросы (в т.ч. на сервере) изолированы.
_ledger: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("clarify_ledger", default=None)

# Канал к человеку: (items) -> list[str] ответов (или None/[] если не отвечает).
# items: [{"question", "options": [...], "why"}]
Clarifier = Callable[[list[dict]], Union[list[str], Awaitable[list[str]]]]
_clarifier: Optional[Clarifier] = None


def set_clarifier(fn: Optional[Clarifier]) -> None:
    """Регистрирует канал уточнений текущего фронтенда (REPL/бот). None → авто-допущения."""
    global _clarifier
    _clarifier = fn


def reset_ledger() -> None:
    """Новый прогон — чистый ledger (зовётся в recall_node)."""
    _ledger.set([])


def ledger() -> list:
    cur = _ledger.get()
    if cur is None:
        cur = []
        _ledger.set(cur)
    return cur


def _assume_value(item: dict) -> str:
    """Разумное допущение: явный assume → первый маркер → общая пометка."""
    if item.get("assume"):
        return item["assume"]
    opts = item.get("options") or []
    return opts[0] if opts else "(выбрано разумное допущение)"


async def ask(items: list[dict]) -> list[dict]:
    """
    Задаёт батч вопросов через зарегистрированный канал, иначе берёт допущения.
    Возвращает (и складывает в ledger) resolved-пункты со status answered|assumed.
    """
    if not items:
        return []
    answers: Optional[list] = None
    if _clarifier is not None:
        try:
            res = _clarifier(items)
            answers = await res if inspect.isawaitable(res) else res
        except Exception:  # noqa: BLE001
            answers = None

    resolved = []
    for i, it in enumerate(items):
        ans = ""
        if answers and i < len(answers) and answers[i]:
            ans = str(answers[i]).strip()
        status = "answered" if ans else "assumed"
        resolved.append({
            "question": it.get("question", ""),
            "options": it.get("options", []),
            "answer": ans or _assume_value(it),
            "status": status,
        })
    ledger().extend(resolved)
    return resolved


def record(question: str, answer: str, status: str = "answered") -> None:
    """Догон: ручная запись Q&A в ledger (из инструмента ask_user)."""
    ledger().append({"question": question, "options": [], "answer": answer, "status": status})


def format_ledger() -> str:
    """Блок для инъекции в промпты нод: что уже уточнено/предположено в этом прогоне."""
    cur = ledger()
    if not cur:
        return "Уточнений по задаче пока нет."
    lines = []
    for it in cur:
        mark = "" if it["status"] == "answered" else "  ⚠ ДОПУЩЕНИЕ (не подтверждено пользователем)"
        lines.append(f"- {it['question']} → {it['answer']}{mark}")
    return "Уточнения и допущения по задаче (учитывай, НЕ переспрашивай заново):\n" + "\n".join(lines)


def has_assumptions() -> bool:
    return any(it["status"] == "assumed" for it in ledger())


async def ask_one(question: str, options: Optional[list[str]] = None, why: str = "") -> str:
    """Догон-обёртка для одного вопроса (используется инструментом ask_user)."""
    resolved = await ask([{"question": question, "options": options or [], "why": why}])
    return resolved[0]["answer"] if resolved else ""


def make_ask_user_tool():
    """
    Инструмент исполнителя: спросить пользователя, когда шаг упёрся в развилку
    (догон). Ответ попадает в общий ledger и переиспользуется следующими шагами.
    """
    from langchain_core.tools import StructuredTool

    async def _ask_user(question: str, options: str = "") -> str:
        # Модели разделяют варианты и обычной «|», и полноширинной «｜» (живой тест:
        # «Да…｜Нет…» слиплось в ОДИН вариант) — нормализуем перед сплитом.
        opts = ([o.strip() for o in options.replace("｜", "|").split("|") if o.strip()]
                if options else [])
        ans = await ask_one(question, opts, why="запрошено исполнителем шага")
        return f"Ответ пользователя: {ans}"

    return StructuredTool.from_function(
        coroutine=_ask_user,
        name="ask_user",
        description=(
            "Спроси пользователя, КОГДА для выполнения шага реально не хватает решения "
            "и его нельзя взять из контекста/памяти (развилка, отсутствует параметр, "
            "несколько равноправных вариантов). options — варианты через '|' (можно пусто "
            "для открытого вопроса). Используй ЭКОНОМНО: сначала ищи ответ сам."
        ),
    )
