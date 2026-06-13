"""
Agentic research-слой: дисциплинированный МНОГОШАГОВЫЙ поиск с ВЕРИФИКАЦИЕЙ промежуточных
фактов — а не наивный ReAct, который часто упирался в бюджет/пасовал на multi-hop.

Цикл (поверх web_search: trafilatura→чанки→BM25S→vector-rerank):
  1. ПЛАН — разбить вопрос на атомарные под-вопросы (искомые факты);
  2. по каждому: поиск → прицельно прочитать топ-источники → ИЗВЛЕЧЬ+ПРОВЕРИТЬ факт
     (found/confidence/источник; не подтверждается текстом → НЕ выдумываем);
  3. накопить ПРОВЕРЕННЫЕ факты;
  4. СИНТЕЗ ответа из проверенного, честные пробелы там, где факт не установлен.

Это даёт grounding (источники), честность (пробелы вместо выдумки) и реальную
многошаговость (найти X → отфильтровать по Y — отдельные под-вопросы).
"""
from __future__ import annotations

import asyncio
import re
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .llm import chat

_URL_RE = re.compile(r"https?://[^\s)\]]+")


class ResearchPlan(BaseModel):
    subquestions: list[str] = Field(
        description="2–5 КОНКРЕТНЫХ под-вопросов — атомарных фактов, которые надо установить из "
                    "веба, чтобы ответить. Для multi-hop (найти X, затем отфильтровать по Y) — "
                    "каждый шаг отдельным под-вопросом, в логическом порядке.")


class FactCheck(BaseModel):
    found: bool = Field(description="Подтверждается ли ДОСТОВЕРНЫЙ ответ на под-вопрос прямо в тексте источника")
    fact: str = Field(description="Извлечённый точный факт (пусто, если не найден)", default="")
    confidence: float = Field(description="Уверенность 0–1", ge=0.0, le=1.0, default=0.0)


_plan_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты планируешь веб-исследование. Разбей вопрос на 2–5 КОНКРЕТНЫХ под-вопросов — "
     "атомарных фактов, которые нужно установить из веба, чтобы ответить. Каждый под-вопрос "
     "= один искомый факт. Для МНОГОШАГОВЫХ задач (найти кандидатов → отфильтровать по "
     "критерию → выбрать) дай шаги в логическом порядке. Не дроби тривиальное."),
    ("human", "Вопрос: {question}"),
])

_verify_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты извлекаешь и ПРОВЕРЯЕШЬ факт из текста источников. found=true ТОЛЬКО если ответ на "
     "под-вопрос ПРЯМО подтверждается текстом. Если в тексте ответа нет — found=false и НЕ "
     "ВЫДУМЫВАЙ (честный пробел лучше галлюцинации). Извлеки точный факт и уверенность.\n"
     "ВНИМАНИЕ на ПОДМЕНУ СУЩНОСТЕЙ: отвечай ИМЕННО на под-вопрос про нужную сущность, не "
     "путай похожие (напр. место проведения события ≠ родина победителя). Учитывай уже "
     "установленные факты как контекст."),
    ("human", "Уже установлено: {known}\n\nПод-вопрос: {subq}\n\nТексты источников:\n{evidence}"),
])

# Зависимая цепочка: под-вопрос с абстрактной ссылкой («столица страны-победителя»)
# переписывается на КОНКРЕТНЫЙ с подстановкой уже найденного («столица Франции») — иначе
# поиск тянет не ту сущность (живой тест: «Москва» вместо «Париж»).
_reformulate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Перепиши под-вопрос, ПОДСТАВИВ конкретные сущности из уже установленных фактов вместо "
     "ссылок-описаний («страна, которая выиграла» → «Франция»). Верни ТОЛЬКО переписанный "
     "под-вопрос, кратко, той же сутью. Если подставлять нечего — верни как есть."),
    ("human", "Установлено: {known}\n\nПод-вопрос: {subq}"),
])

# Рефлексия-ретрай (паттерн «Observation-Reflection»): если под-вопрос не нашёлся —
# одна попытка с АЛЬТЕРНАТИВНОЙ формулировкой (другие ключевые слова/угол). Тратит вызовы
# только на неудаче, повышает завершаемость multi-hop цепочки.
_alt_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Прошлый поисковый запрос НЕ дал ответа на под-вопрос. Сформулируй ОДИН АЛЬТЕРНАТИВНЫЙ "
     "веб-запрос для того же факта — другие ключевые слова/угол/синонимы. Верни ТОЛЬКО запрос."),
    ("human", "Под-вопрос: {subq}\nУже известно: {known}"),
])


_synth_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Собери ответ на ИСХОДНЫЙ вопрос. Приоритет — ПРОВЕРЕННЫЕ факты. Если их не хватает — "
     "посмотри СЫРЫЕ НАХОДКИ (реальные результаты поиска и текст страниц) и ИЗВЛЕКИ ответ ИЗ "
     "НИХ: это чтение реальных источников, не выдумка. НЕ отвечай «невозможно определить», если "
     "ответ есть в находках — извлеки его (посчитай/перечисли/назови). Выдумывать СВЕРХ "
     "найденного нельзя: если ответа реально нет ни в фактах, ни в находках — честно скажи, "
     "чего не хватает. Ответ по существу, кратко, на языке вопроса; где уместно — источник."),
    ("human", "Вопрос: {question}\n\nПроверенные факты:\n{facts}\n\nСырые находки (реальные источники):\n{evidence}"),
])


async def agentic_research(question: str, max_subq: int = 4, max_sources: int = 2,
                           deadline: float = 90.0) -> dict:
    """
    Многошаговый research с верификацией. САМООГРАНИЧЕН ПО ВРЕМЕНИ (deadline сек): между
    под-вопросами проверяет остаток и при нехватке СТОП → синтез из уже проверенного (не
    бросает таймаут, не даёт удвоения через ReAct-фолбэк). Возвращает {answer,facts,verified,total}.
    """
    from src.skills.web_search.web_search import search_web, browse
    _t0 = time.monotonic()

    def _left() -> float:
        return deadline - (time.monotonic() - _t0)

    fast = chat("fast", 0)        # план/реформуляция — дёшево, это «роутинг»
    try:
        plan = await (_plan_prompt | fast.with_structured_output(ResearchPlan)).ainvoke({"question": question})
        subqs = [s for s in (plan.subquestions or []) if s.strip()][:max_subq] or [question]
    except Exception:  # noqa: BLE001
        subqs = [question]

    # БЫСТРОЕ чтение БЕЗ cloakbrowser: urllib+trafilatura + наш chunk-экстракт. cloak спавнит
    # Chromium-подпроцесс (35с/страница, asyncio _do_waitpid висел) — для фактосбора не нужен.
    from src.skills.web_search.web_search import _page_text_urllib, _relevant_chunks

    def _fast_read(url: str, find: str) -> str:
        title, text = _page_text_urllib(url)
        if not text:
            return ""
        return f"# {title}\n" + _relevant_chunks(text, find, budget=3000)

    async def _resolve(sq_q: str, known: str) -> dict:
        """Один проход: поиск → БЫСТРОЕ прицельное чтение → ИЗВЛЕЧЬ+ПРОВЕРИТЬ факт."""
        try:
            # КАП на поиск: search_web падает в cloakbrowser (Chromium, 35с) когда SearXNG лёг —
            # в research это ×под-вопросы = взрыв. 12с хватает на SearXNG/urllib, cloak бросаем.
            res = await asyncio.wait_for(
                asyncio.to_thread(search_web.invoke, {"query": sq_q, "max_results": 4}), timeout=12)
        except Exception as e:  # noqa: BLE001
            return {"subq": sq_q, "found": False, "fact": f"(поиск прерван: {type(e).__name__})", "conf": 0.0, "sources": []}
        urls = _URL_RE.findall(res)[:max_sources]
        # СНИППЕТЫ ПОИСКА как evidence: выдача (заголовки+сниппеты) часто УЖЕ содержит ответ —
        # робастно даже когда страница не читается (IMDB/JS блокируют чтение). Берём всегда.
        snippets = "[Выдача поиска]\n" + res[:2000]
        # ПАРАЛЛЕЛЬНОЕ чтение источников (было последовательно ×15с) — режет время под-вопроса.
        reads = await asyncio.gather(
            *[asyncio.wait_for(asyncio.to_thread(_fast_read, u, sq_q), timeout=15) for u in urls],
            return_exceptions=True)
        pages = "\n\n".join(r for r in reads if isinstance(r, str) and r.strip())[:5000]
        evidence = (snippets + ("\n\n" + pages if pages else "")).strip()
        if not evidence:
            return {"subq": sq_q, "found": False, "fact": "(нет результатов поиска)", "conf": 0.0, "sources": urls, "evidence": ""}
        try:
            fc = await (_verify_prompt | fast.with_structured_output(FactCheck)).ainvoke(
                {"subq": sq_q, "known": known or "(пока ничего)", "evidence": evidence[:6000]})
            return {"subq": sq_q, "found": fc.found, "fact": fc.fact, "conf": fc.confidence,
                    "sources": urls, "evidence": evidence[:2500]}
        except Exception:  # noqa: BLE001
            return {"subq": sq_q, "found": False, "fact": "(не удалось проверить)", "conf": 0.0,
                    "sources": urls, "evidence": evidence[:2500]}

    facts: list[dict] = []
    for sq in subqs:
        if _left() < 18:  # времени на ещё один под-вопрос нет → СТОП, синтезируем что есть
            break
        # ЗАВИСИМАЯ ЦЕПОЧКА: подставляем уже найденное в абстрактные ссылки под-вопроса.
        known = "; ".join(f["fact"] for f in facts if f["found"] and f["conf"] >= 0.5)
        sq_q = sq
        if known:
            try:
                rf = await (_reformulate_prompt | fast).ainvoke({"known": known, "subq": sq})
                sq_q = (rf.content if hasattr(rf, "content") else str(rf)).strip() or sq
            except Exception:  # noqa: BLE001
                sq_q = sq
        fact = await _resolve(sq_q, known)
        # РЕФЛЕКСИЯ-РЕТРАЙ: не нашли → одна попытка с альтернативной формулировкой (если есть время).
        if not fact["found"] and _left() > 22:
            try:
                alt = await (_alt_prompt | fast).ainvoke({"subq": sq_q, "known": known or "(нет)"})
                alt_q = (alt.content if hasattr(alt, "content") else str(alt)).strip()
                if alt_q and alt_q.lower() != sq_q.lower():
                    fact2 = await _resolve(alt_q, known)
                    if fact2["found"]:
                        fact = fact2
            except Exception:  # noqa: BLE001
                pass
        facts.append(fact)

    verified = [f for f in facts if f["found"] and f["conf"] >= 0.5]
    facts_text = "\n".join(
        f"- {f['subq']} → {f['fact']} (уверенность {f['conf']:.0%}"
        + (f", источник: {f['sources'][0]}" if f['sources'] else "") + ")"
        for f in verified) or "(достоверных фактов не установлено)"
    # АНТИ-«сдался»: если проверенных фактов МЕНЬШЕ, чем под-вопросов, отдаём синтезу СЫРЫЕ
    # НАХОДКИ (реальные сниппеты/текст страниц) — пусть извлечёт ответ ИЗ НИХ (это чтение
    # реальных источников, не выдумка), а не отвечает «невозможно определить» при наличии данных.
    # Живой баг GAIA L1 (Mercedes Sosa albums): research сдавался, хотя ответ был в находках.
    evidence_blob = "(находок нет)"
    if len(verified) < len(facts):
        ev = "\n\n".join(f"[{f['subq']}]\n{f.get('evidence', '')}" for f in facts if f.get("evidence"))
        evidence_blob = ev[:6000] or "(находок нет)"
    try:
        ans = await (_synth_prompt | fast).ainvoke(
            {"question": question, "facts": facts_text, "evidence": evidence_blob})
        answer = ans.content if hasattr(ans, "content") else str(ans)
    except Exception:  # noqa: BLE001
        answer = facts_text
    return {"answer": answer, "facts": facts, "verified": len(verified), "total": len(facts)}


def _deep_research(question: str) -> str:
    """Sync-обёртка тула: запускает agentic_research в своём event loop (вызывается из to_thread)."""
    res = asyncio.run(agentic_research(question))
    head = f"[research: {res['verified']}/{res['total']} под-вопросов подтверждено]\n"
    return head + res["answer"]


def make_deep_research_tool() -> StructuredTool:
    class _Q(BaseModel):
        question: str = Field(description="Сложный многошаговый вопрос для веб-исследования "
                                          "(найти→отфильтровать→сопоставить факты)")

    async def _arun(question: str) -> str:
        res = await agentic_research(question)
        head = f"[research: {res['verified']}/{res['total']} под-вопросов подтверждено]\n"
        return head + res["answer"]

    return StructuredTool.from_function(
        coroutine=_arun, name="deep_research", args_schema=_Q,
        description="DEEP multi-hop web research with intermediate-fact VERIFICATION. Use for hard "
                    "questions needing several lookups chained (find candidates → filter by a "
                    "criterion → cross-reference). Returns a grounded answer with honest gaps "
                    "where facts couldn't be verified — better than ad-hoc search for such tasks.",
    )
