"""
ЭКСПЕРИМЕНТАЛЬНЫЙ граф (флаг `experimental.composer`): мета-контроллер `composer` компонует
когнитивные примитивы под задачу вместо выбора «1 из 6 запечённых режимов». Плюс раздел M:
само-модель как ПРИОР композиции + verify-гейт + session-commit (см. `src/runtime/self_model.py`).

ДОЛГ КОПИИ УСТРАНЁН (2026-06-26): раньше файл был почти-полной копией `agent.py` (52/61 функций,
22 из них УСПЕЛИ отстать → фиксы приходилось зеркалить вручную). Теперь это ТОНКОЕ расширение —
все узлы/хелперы/чейны/константы берутся из ЖИВОГО графа (`agent.py` = единственный источник
правды), здесь определяется ТОЛЬКО уникальное: композитор и его примитивы. Любой фикс в `agent.py`
(recall/synthesize/validation/…) теперь автоматически доезжает сюда.

Глобалы, которые переинициализируются (`rebuild_llms` → llm/code_llm/deep_llm/чейны), читаются
через `_live.<имя>`, а не биндятся локально — иначе после смены провайдера композитор держал бы
устаревшие объекты. Живой граф (`agent.py`) этим файлом НЕ затрагивается.
"""
from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field as _Field
from langchain_core.messages import SystemMessage, HumanMessage

from src.graph import agent as _live           # живой граф — единый источник правды
from src.graph.agent import *                   # noqa: F401,F403 — узлы/хелперы/чейны/константы/чейны
from src.graph.agent import (                   # приватные хелперы (import * их не берёт)
    _exec_compose, _exec_direct, _skills_for_act,
)
from src.graph.schemas import GeneralGraphState
from src.runtime import self_model, run_context

# ── мета-контроллер композиции примитивов (вместо «роутер выбирает 1 из 6 режимов») ──────────
# На КАЖДОМ шаге контроллер выбирает СЛЕДУЮЩИЙ примитив, глядя на накопленное, и сам решает, когда
# финализировать. Режимы fast/reason/act/… — частные «запечённые» рецепты этой композиции.
EXP_PRIMITIVES = ("recall", "reason", "act", "verify", "finalize")


class _ComposerStep(BaseModel):
    """Решение мета-контроллера: какой ОДИН примитив запустить следующим."""
    primitive: str = _Field(description="recall|reason|act|verify|finalize")
    goal: str = _Field(default="", description="конкретная под-задача для этого примитива")
    done: bool = _Field(default=False, description="данных достаточно для финала")
    rationale: str = _Field(default="", description="кратко почему этот шаг")


class _VerifyVerdict(BaseModel):
    """Структурный вердикт verify — чтобы он ГЕЙТИЛ финализацию (раздел M0), а не был театром."""
    ok: bool = _Field(description="данных достаточно и нет противоречий")
    missing: str = _Field(default="", description="чего не хватает / что под вопросом, кратко")


_COMPOSER_SYS = (
    "Ты — мета-контроллер. Решаешь, какой ОДИН когнитивный примитив запустить "
    "следующим, чтобы продвинуть задачу. Примитивы:\n"
    "- recall — достать из памяти юзера релевантные факты/прошлые решения;\n"
    "- reason — подумать шаг БЕЗ инструментов (разложить, вывести из собранного);\n"
    "- act — ОДНО действие инструментами (веб-поиск, браузер, вычисление, файл);\n"
    "- verify — проверить текущие факты/черновик на противоречия и пробелы;\n"
    "- finalize — собрать финальный ответ (когда данных достаточно).\n"
    "Смотри на ИСХОДНЫЙ запрос и НАКОПЛЕННЫЕ результаты. Не повторяй сделанное. "
    "Экономь шаги: как только ответа достаточно — primitive=finalize или done=true."
)


def _scratch_text(scratch: list[dict]) -> str:
    if not scratch:
        return "(пока пусто)"
    return "\n\n".join(
        f"[{i + 1}] {s['primitive']} · цель: {s.get('goal', '')}\n→ {(s.get('output') or '')[:800]}"
        for i, s in enumerate(scratch))


async def _composer_pick(query: str, mem: str, scratch: list[dict], steps_left: int) -> _ComposerStep:
    sys = _COMPOSER_SYS + f"\n\nОсталось шагов: {steps_left}."
    ctx = (f"ИСХОДНЫЙ ЗАПРОС:\n{query}\n\nПАМЯТЬ:\n{mem or '—'}\n\n"
           f"НАКОПЛЕНО:\n{_scratch_text(scratch)}\n\nКакой примитив следующий?")
    try:
        d = await _live.llm.with_structured_output(_ComposerStep).ainvoke(
            [SystemMessage(content=sys), HumanMessage(content=ctx)])
        if d.primitive not in EXP_PRIMITIVES:
            d.primitive = "reason"
        return d
    except Exception as e:  # noqa: BLE001
        return _ComposerStep(primitive="finalize", done=True,
                             rationale=f"controller error: {type(e).__name__}")


async def _prim_recall(goal: str, state: GeneralGraphState) -> str:
    try:
        uid = state.get("user_id", "default")
        ms = _live.memory_store
        qv = await ms.embedder.aembed(goal) if ms.embedder.enabled else None
        txt, _ = ms.recall_scored(uid, goal, qvec=qv)
        return txt or "(в памяти ничего релевантного)"
    except Exception as e:  # noqa: BLE001
        return f"(recall недоступен: {type(e).__name__})"


async def _prim_reason(goal: str, query: str, scratch: list[dict], deadline: float) -> str:
    sys = ("Ты — аккуратный мыслитель. Сделай ОДИН содержательный шаг рассуждения под "
           "под-задачу, опираясь ТОЛЬКО на исходный запрос и уже собранные результаты. Без выдумок.")
    prompt = f"ЗАПРОС: {query}\n\nСОБРАНО:\n{_scratch_text(scratch)}\n\nПОД-ЗАДАЧА: {goal}"
    out, _ = await _exec_compose(sys, prompt, deadline)
    return out


async def _prim_act(goal: str, state: GeneralGraphState, qe, deadline: float) -> str:
    picked = _skills_for_act(goal, qvec=qe)
    tools = get_all_loaded_skill_tools(picked)
    if not tools:
        return "(нет подходящих инструментов под это действие)"
    tools.append(clarify.make_ask_user_tool())
    sys = act_system_prompt.format(memory_context=state.get("memory_context", "Память пуста."))
    try:
        out, _ = await _exec_direct(sys, goal, tools, deadline)
        return out
    except Exception as e:  # noqa: BLE001
        return f"(действие не удалось: {type(e).__name__})"


async def _prim_verify(goal: str, query: str, scratch: list[dict], deadline: float):
    """Верификатор со СТРУКТУРНЫМ вердиктом — verify ГЕЙТИТ финализацию (раздел M0): пока ok=False,
    composer не вправе финализировать. Возвращает (текст, ok, missing). fail-open на ошибке —
    чтобы сбой verify не задедлочил цикл навечно."""
    sys = ("Ты — критик-верификатор. Проверь собранные результаты на противоречия, пробелы и "
           "невалидные утверждения относительно запроса.")
    prompt = f"ЗАПРОС: {query}\n\nСОБРАНО:\n{_scratch_text(scratch)}\n\nФОКУС ПРОВЕРКИ: {goal}"
    try:
        v = await asyncio.wait_for(
            _live.llm.with_structured_output(_VerifyVerdict).ainvoke(
                [SystemMessage(content=sys), HumanMessage(content=prompt)]),
            timeout=max(8.0, deadline))
        text = "verify: OK" if v.ok else ("verify: ПРОБЕЛ — " + (v.missing or "не хватает данных"))
        return text, bool(v.ok), (v.missing or "")
    except Exception as e:  # noqa: BLE001
        return f"(verify недоступен: {type(e).__name__})", True, ""


async def _composer_finalize(query: str, mem: str, scratch: list[dict], deadline: float) -> str:
    sys = ("Собери ФИНАЛЬНЫЙ ответ пользователю строго из накопленных результатов. Не выдумывай "
           "фактов и ссылок. Если данных не хватило — честно скажи, что удалось, а что нет.")
    prompt = (f"ЗАПРОС: {query}\n\nПАМЯТЬ:\n{mem or '—'}\n\n"
              f"СОБРАННЫЕ РЕЗУЛЬТАТЫ:\n{_scratch_text(scratch)}")
    out, _ = await _exec_compose(sys, prompt, deadline)
    return out


async def composer_node(state: GeneralGraphState) -> dict:
    """ЭКСПЕРИМЕНТ (раздел M): мета-контроллер компонует примитивы. На старте — пробуждение:
    env-id (изоляция по умолчанию) + детерминированный self-model как ПРИОР композиции. verify
    ГЕЙТИТ финализацию. По завершении — session-commit «что сделано в этом проекте»."""
    query = state["query"]
    mem = state.get("memory_context", "")
    qe = state.get("query_emb") or None
    max_steps = int(_live.config.get("experimental", {}).get("max_steps", 12))

    # ── M0: пробуждение — env-id + детерминированный self-model (без новых LLM-вызовов) как приор ──
    env_id = self_model.resolve_env_id(
        state.get("surface", "chat"), state.get("session_id", ""),
        run_context.current_run_id() or "")
    sm = self_model.build_self_model(
        env_id=env_id, store=_live.memory_store, user_id=state.get("user_id", ""),
        query=query, qvec=qe, primitives=EXP_PRIMITIVES)
    prior = (sm + "\n\n" + mem) if sm else mem      # self-model впереди памяти → приор композиции

    scratch: list[dict] = []
    trace: list[str] = []
    verify_pending = False                          # незакрытый пробел из verify запрещает finalize
    last_gap = ""
    for i in range(max_steps):                      # range(max_steps) — структурный пол (анти-баг)
        deadline = _live.STEP_DEADLINE_CAP  # анти-зависание вызова, без времячувствительности прогона
        d = await _composer_pick(query, prior, scratch, max_steps - i)
        # verify-ГЕЙТ: пока пробел открыт — финал нельзя, форсим его закрытие
        if (d.primitive == "finalize" or d.done) and verify_pending:
            d.primitive, d.done = "reason", False
            d.goal = d.goal or ("закрыть пробел из verify: " + last_gap)
            trace.append("gate")
        trace.append(f"{d.primitive}({(d.goal or '')[:32]})")
        if d.primitive == "finalize" or d.done:
            break
        if d.primitive == "recall":
            out = await _prim_recall(d.goal or query, state)
        elif d.primitive == "act":
            out = await _prim_act(d.goal or query, state, qe, deadline)
        elif d.primitive == "verify":
            out, ok, last_gap = await _prim_verify(d.goal, query, scratch, deadline)
            verify_pending = not ok
        else:  # reason
            out = await _prim_reason(d.goal or query, query, scratch, deadline)
        scratch.append({"primitive": d.primitive, "goal": d.goal, "output": out})
    deadline = _live.STEP_DEADLINE_CAP  # анти-зависание вызова, без времячувствительности прогона
    answer = await _composer_finalize(query, prior, scratch, deadline)
    grounded = query + "\n" + (mem or "") + "\n" + _scratch_text(scratch)
    answer = strip_ungrounded_pii(answer, grounded)
    print(f"[Composer] env={env_id} {' → '.join(trace)}")
    # ── закрытие сессии: session-commit «что сделано» (фоново-безопасно, не валит ответ) ──
    try:
        self_model.session_commit(
            env_id, queries=1, primitives=[s["primitive"] for s in scratch],
            skills_used=[s.get("goal", "")[:40] for s in scratch if s["primitive"] == "act"])
    except Exception:  # noqa: BLE001
        pass
    return {"final_answer": answer, "mode": "composer",
            "mode_rationale": "композиция примитивов: " + " → ".join(trace)}


def build_graph(checkpointer=None):
    """ЭКСПЕРИМЕНТАЛЬНЫЙ граф: recall → composer (мета-контроллер примитивов) → reflect.
    Память (recall) и обучение (reflect) переиспользуются из ЖИВОГО графа; заменён лишь весь
    средний слой выбора режима — его собирает composer_node на ходу."""
    graph = StateGraph(GeneralGraphState)
    for _name, _fn in {"recall": recall_node, "composer": composer_node,
                        "reflect": reflect_node}.items():
        graph.add_node(_name, traced(_name, _fn))
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "composer")
    graph.add_edge("composer", "reflect")
    graph.add_edge("reflect", END)
    return graph.compile(checkpointer=checkpointer)


# Граф «1 из 6 режимов» для сравнения (composer vs modes) = ЖИВОЙ граф (единый источник правды,
# больше не дублируется здесь — это и был долг копии).
_build_graph_modes = _live.build_graph
