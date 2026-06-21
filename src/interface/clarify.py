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

import inspect
import re
from typing import Awaitable, Callable, Optional, Union

# Ledger одного прогона — ОБЩИЙ dict по run_id (run_id с ГРАНИЦЫ запроса, run_context). Раньше был
# contextvar, выставляемый в recall_node — но set-в-ноде НЕ виден сёстрам-нодам/ретраям (−9pp-капкан):
# на ретрае step_executor ledger лениво создавался ПУСТЫМ → дедуп не видел прошлый ответ → clarify
# ПЕРЕСПРАШИВАЛ на каждом «step failed — retrying» (баг ревью). dict по run_id виден на ВСЕХ нодах/
# ретраях прогона; изоляция мульти-клиента — по ключу; без run_scope работает общий "_default".
from src.runtime import run_context as _rc
_ledgers: dict[str, list] = {}
_rc.register_cleanup(lambda rid: _ledgers.pop(rid, None))


def _key() -> str:
    return _rc.current_run_id() or "_default"

# Канал к человеку: (items) -> list[str] ответов (или None/[] если не отвечает).
# items: [{"question", "options": [...], "why"}]
Clarifier = Callable[[list[dict]], Union[list[str], Awaitable[list[str]]]]
_clarifier: Optional[Clarifier] = None


def set_clarifier(fn: Optional[Clarifier]) -> None:
    """Регистрирует канал уточнений текущего фронтенда (REPL/бот). None → авто-допущения."""
    global _clarifier
    _clarifier = fn


def reset_ledger() -> None:
    """Новый прогон — чистый ledger (зовётся в recall_node). Ключ — run_id с границы запроса."""
    _ledgers[_key()] = []


def ledger() -> list:
    k = _key()
    cur = _ledgers.get(k)
    if cur is None:
        cur = []
        _ledgers[k] = cur
    return cur


def _assume_value(item: dict) -> str:
    """Разумное допущение: явный assume → первый маркер → общая пометка."""
    if item.get("assume"):
        return item["assume"]
    opts = item.get("options") or []
    return opts[0] if opts else "(выбрано разумное допущение)"


def _norm_q(q: str) -> str:
    """Нормализация вопроса для дедупа (регистр/пунктуация/пробелы не важны)."""
    return re.sub(r"\W+", " ", (q or "").lower()).strip()


async def ask(items: list[dict]) -> list[dict]:
    """
    Задаёт батч вопросов через зарегистрированный канал, иначе берёт допущения.
    Возвращает (и складывает в ledger) resolved-пункты со status answered|assumed.
    ДЕДУП: вопросы, уже отвеченные в этом прогоне (clarify_gate → потом ask_user в шаге),
    НЕ переспрашиваются — берём ответ из ledger (status=reused). Иначе агент дублировал
    уточнения: сначала полный батч, потом по каждому пункту отдельно.
    """
    if not items:
        return []
    prior = {_norm_q(e["question"]): e for e in ledger()
             if e.get("status") in ("answered", "reused") and e.get("answer")}
    ask_now = [it for it in items if _norm_q(it.get("question", "")) not in prior]

    fresh: dict[str, str] = {}
    if ask_now and _clarifier is not None:
        try:
            res = _clarifier(ask_now)
            answers = await res if inspect.isawaitable(res) else res
        except Exception:  # noqa: BLE001
            answers = None
        for j, it in enumerate(ask_now):
            a = answers[j] if answers and j < len(answers) else None
            fresh[_norm_q(it.get("question", ""))] = str(a).strip() if a else ""

    resolved = []
    for it in items:
        nq = _norm_q(it.get("question", ""))
        if nq in prior:  # уже спрашивали в этом прогоне → переиспользуем, НЕ дублируем
            resolved.append({"question": it.get("question", ""), "options": it.get("options", []),
                             "answer": prior[nq]["answer"], "status": "reused"})
            continue
        ans = fresh.get(nq, "")
        resolved.append({
            "question": it.get("question", ""),
            "options": it.get("options", []),
            "answer": ans or _assume_value(it),
            "status": "answered" if ans else "assumed",
        })
    ledger().extend([r for r in resolved if r["status"] != "reused"])
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
