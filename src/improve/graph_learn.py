"""
Graph-aware batch self-improvement — «обучение» графа агента.

Идея: граф = дифференцируемая программа. Трейс одного прогона = forward pass
(какие ноды активировались). Backward pass = credit assignment: по батчу неудач
смотрим, какие ноды чаще всего фигурировали в провальных прогонах («слабые связи»),
и батч-оптимизируем параметры самой виноватой оптимизируемой ноды.

Генерализация: чем больше батч (прогонов на «обучение»), тем устойчивее
закономерность вины и тем меньше подгонка под единичный случай.

Сейчас оптимизируемый артефакт-промпт есть у нод с plain-string промптом
(step_executor). По мере перевода остальных нод на override-строки карта
OPTIMIZABLE расширяется — и backward становится по-настоящему graph-wide.
"""
from __future__ import annotations

import os
from collections import Counter

from omegaconf import OmegaConf

from ..memory import MemoryStore
from ..prompts import backward_prompt
from ..structured_outputs import NodeGradients
from ..tracing import trace_store
from .pipe import SelfLearningPipe

_cfg = OmegaConf.load("config.yml")

NODE_DESC = {
    "goal": "определяет цель и стоящую цель/rubric",
    "reflexion": "выбирает режим мышления (fast/reason/deliberate/heavy/clarify)",
    "decompose": "раскладывает задачу на подшаги с done_check",
    "fast_answer": "быстрый интуитивный ответ без инструментов",
    "reason": "глубокое пошаговое рассуждение без инструментов",
    "step_executor": "исполняет подшаг инструментами и самопроверяет",
    "review": "heavy: сквозной ревью собранного решения, выдаёт fix-подшаги",
    "clarify_gate": "батч уточнений перед исполнением неясной задачи",
}


def _backward_gradients(blamed: list[str], failures_text: str):
    """ОДИН LLM-вызов: textual gradients по виноватым нодам, рассуждая вдоль forward-цепочек."""
    from ..llm import chat, provider

    if provider() == "openrouter" and not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
        return []
    llm = chat("fast", 0)
    catalog = "\n".join(f"- {n} ({OPTIMIZABLE[n]}): {NODE_DESC.get(n, '')}" for n in blamed)
    chain = backward_prompt | llm.with_structured_output(NodeGradients)
    try:
        res = chain.invoke({"node_catalog": catalog, "blamed": ", ".join(blamed), "failures": failures_text})
        return res.gradients
    except Exception as e:  # noqa: BLE001
        print(f"[graph_backward] gradient call failed: {e}")
        return []


def _format_failure_chains(fails: list, failures: list[dict]) -> str:
    """Готовит текст неудач с forward-цепочкой (нода→выход) из трейса для edge-gradient."""
    blocks = []
    for i, (row, f) in enumerate(zip(fails[:12], failures[:12]), 1):
        rid = row["run_id"] if "run_id" in row.keys() else ""
        chain = trace_store.run_trace(rid) if rid else []
        chain_txt = "  " + " → ".join(f"{n}[{(o or '')[:50]}]" for n, o in chain) if chain else "  (трейс недоступен)"
        blocks.append(
            f"[{i}] Запрос: {f['query'][:160]}\n  Финал: {f['answer'][:160]}\n"
            f"  Замечание: {f.get('feedback', '') or '(нет)'}\n  Цепочка: {chain_txt}"
        )
    return "\n\n".join(blocks)

def _is_poisoned(f) -> bool:
    """Эпизод-неудача отравлен инъекцией? Смотрим И query, И answer (инъекция могла прийти через
    вывод тула и отравить траекторию). Real backward (graph_backward / graph_backward_user) ходит
    через ЭТО, а не через query-only is_unsafe_to_learn (долг ревью B3)."""
    from .safety import is_unsafe_to_learn
    try:
        ans = f["answer"] if "answer" in f.keys() else ""
    except Exception:  # noqa: BLE001
        ans = ""
    return is_unsafe_to_learn(f["query"]) or is_unsafe_to_learn(ans)


# Нода графа → роль оптимизируемого промпта (artifact в ParamStore).
# Теперь graph-wide: каждая когнитивная нода — обучаемый параметр.
OPTIMIZABLE = {
    "goal": "goal",
    "reflexion": "reflexion",
    "decompose": "decompose",
    "fast_answer": "fast_answer",
    "reason": "reason",
    "step_executor": "step_execution",
    "review": "review",
    "clarify_gate": "clarify_gate",
}


def _node_rates(runs: dict[str, list[str]]) -> dict[str, float]:
    """Доля прогонов, в которых активировалась оптимизируемая нода."""
    total = len(runs) or 1
    cnt: Counter = Counter()
    for nodes in runs.values():
        for node in set(nodes):
            if node in OPTIMIZABLE:
                cnt[node] += 1
    return {node: cnt[node] / total for node in cnt}


def credit_assignment(memory_store: MemoryStore, min_batch: int) -> tuple[dict, list[dict]]:
    """
    Backward с ДИФФЕРЕНЦИАЛЬНОЙ виной: нода виновата, если активируется в неудачах
    ЧАЩЕ, чем в успехах (blame = failRate − successRate). Так goal/reflexion,
    срабатывающие всегда, не получают ложную вину — выделяются реально слабые связи.
    """
    # Не учимся на попытках взлома собственной защиты (инъекции/джейлбреки исключаются
    # ДО анализа вины — иначе backward мог бы «оптимизировать» обход guardrails). Проверяем И query,
    # И answer (инъекция могла прийти через вывод тула и отравить траекторию — B3).
    fails = [f for f in memory_store.get_failures(n=60) if not _is_poisoned(f)]
    if len(fails) < min_batch:
        return {}, list(fails)
    sucs = memory_store.get_successes(n=40)

    def rid(e):
        return e["run_id"] if "run_id" in e.keys() else ""

    fail_runs = trace_store.nodes_for_runs([rid(f) for f in fails if rid(f)])
    suc_runs = trace_store.nodes_for_runs([rid(s) for s in sucs if rid(s)])

    fr, sr = _node_rates(fail_runs), _node_rates(suc_runs)
    blame = {node: round(fr.get(node, 0.0) - sr.get(node, 0.0), 3) for node in OPTIMIZABLE}
    blame = {n: v for n, v in blame.items() if v > 0}  # только «чаще в неудачах»

    if not blame:  # нет трейс-инфо (старые эпизоды) → дефолт на исполнителя
        blame = {"step_executor": 1.0}
    return blame, list(fails)


def graph_backward(memory_store: MemoryStore, min_batch: int = 6, accept: bool = True, max_nodes: int = 3) -> dict:
    """
    Полный backward по графу: дифф-вина → per-node textual gradients (1 LLM-вызов) →
    оптимизация КАЖДОЙ виноватой ноды её собственным градиентом (multi-node).
    """
    blame, fails = credit_assignment(memory_store, min_batch)
    if len(fails) < min_batch:
        return {"status": "skipped", "reason": f"мало неудач ({len(fails)}/{min_batch})"}
    blamed = sorted(blame, key=blame.get, reverse=True)[:max_nodes]
    if not blamed:
        return {"status": "skipped", "reason": "нет виноватых нод"}

    # Политика: системные промпты ключевых нод заморожены. Если все виноватые ноды —
    # core (промпт не переписываем), не тратим LLM-вызов на градиенты: их канал
    # улучшения — few-shots (forward-харвест), не backward-переписывание промпта.
    from .pipe import _prompt_tunable

    tunable_blamed = [n for n in blamed if _prompt_tunable(OPTIMIZABLE.get(n, n))]
    if not tunable_blamed:
        return {"status": "skipped", "reason": "виноватые ноды — ключевые (промпты заморожены); "
                "канал улучшения для них — few-shots", "blame": blame, "blamed_nodes": blamed}

    failures = [
        {"query": f["query"], "answer": f["answer"], "feedback": (f["feedback"] if "feedback" in f.keys() else "")}
        for f in fails
    ]
    # edge-aware backward: даём модели forward-цепочки (нода→выход) из трейса
    failures_text = _format_failure_chains(fails, failures)
    gradients = _backward_gradients(tunable_blamed, failures_text)

    pipe = SelfLearningPipe(memory_store)
    results = []
    if gradients:  # per-node textual gradient
        for g in gradients:
            if g.node in OPTIMIZABLE:
                results.append(pipe.optimize_role(OPTIMIZABLE[g.node], failures, gradient=g.critique, accept=accept))
    else:  # фолбэк без градиентов — оптимизируем самую виноватую
        results.append(pipe.optimize_role(OPTIMIZABLE[blamed[0]], failures, accept=accept))

    return {
        "status": "done",
        "blame": blame,
        "blamed_nodes": blamed,
        "batch_size": len(fails),
        "gradients": [{"node": g.node, "critique": g.critique[:120]} for g in gradients],
        "results": results,
    }


# Обратная совместимость: batch_optimize == graph_backward.
def batch_optimize(memory_store: MemoryStore, min_batch: int = 6, accept: bool = True) -> dict:
    return graph_backward(memory_store, min_batch=min_batch, accept=accept)


def _user_blame(memory_store: MemoryStore, user_id: str, fails: list) -> dict:
    """Вина нод на прогонах ИМЕННО этого пользователя (failRate − successRate)."""
    sucs = memory_store.get_successes(n=40, user_id=user_id)

    def rid(e):
        return e["run_id"] if "run_id" in e.keys() else ""

    fr = _node_rates(trace_store.nodes_for_runs([rid(f) for f in fails if rid(f)]))
    sr = _node_rates(trace_store.nodes_for_runs([rid(s) for s in sucs if rid(s)]))
    blame = {n: round(fr.get(n, 0.0) - sr.get(n, 0.0), 3) for n in OPTIMIZABLE}
    blame = {n: v for n, v in blame.items() if v > 0}
    return blame or {"step_executor": 1.0}


def graph_backward_user(memory_store: MemoryStore, user_id: str, min_batch: int = 3,
                        accept: bool = True, max_nodes: int = 3) -> dict:
    """
    PER-USER backward — СЕРДЦЕ «оптимизации под пользователя» (полезен ИМЕННО этому человеку).
    Из НЕУДАЧ конкретного юзера + того, КТО он (профиль/роли), синтезирует корректирующие
    УРОКИ и кладёт их как ПЕРСОНАЛЬНЫЕ few-shots для виноватых нод. Ядро НЕ трогаем
    (системные промпты заморожены) — few-shots это разрешённый обратимый канал. Пишет ТОЛЬКО
    в user_fewshots этого пользователя (изоляция). Безопасность: джейлбрейки исключены из батча.
    """
    import os as _os

    from ..llm import chat, provider
    from ..prompts import user_backward_prompt
    from ..structured_outputs import UserLessons
    from .prompt_store import add_user_fewshot
    from .safety import is_unsafe_to_learn

    if not user_id:
        return {"status": "skipped", "reason": "нет user_id"}
    if provider() == "openrouter" and not (_os.getenv("OPEN_ROUTER_API_KEY") or _os.getenv("OPENAI_API_KEY")):
        return {"status": "error", "reason": "нет API-ключа"}

    fails = [f for f in memory_store.get_failures(n=40, user_id=user_id) if not _is_poisoned(f)]
    if len(fails) < min_batch:
        return {"status": "skipped", "reason": f"мало неудач у пользователя ({len(fails)}/{min_batch})"}

    blame = _user_blame(memory_store, user_id, fails)
    blamed = sorted(blame, key=blame.get, reverse=True)[:max_nodes]
    profile = memory_store.format_profile(user_id) or "Профиль пуст."
    failures = [
        {"query": f["query"], "answer": f["answer"], "feedback": (f["feedback"] if "feedback" in f.keys() else "")}
        for f in fails
    ]
    failures_text = _format_failure_chains(fails, failures)
    catalog = "\n".join(f"- {n} ({OPTIMIZABLE[n]}): {NODE_DESC.get(n, '')}" for n in blamed)

    chain = user_backward_prompt | chat("fast", 0).with_structured_output(UserLessons)
    try:
        lessons = chain.invoke({"profile": profile, "node_catalog": catalog,
                                "blamed": ", ".join(blamed), "failures": failures_text}).lessons
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "reason": f"уроки не сгенерированы: {e}"}

    stored = []
    if accept:
        for l in lessons:
            if l.node in OPTIMIZABLE:
                # урок → персональный few-shot нужной ноды (high score: приоритетнее общих).
                # kind='lesson' → рендерится ОТДЕЛЬНЫМ блоком, не подменяет mode-метку (B6).
                add_user_fewshot(user_id, OPTIMIZABLE[l.node], l.trigger, l.lesson, score=0.9, kind="lesson")
                stored.append({"node": l.node, "trigger": l.trigger[:60]})
    return {"status": "done", "user_id": user_id, "blame": blame, "blamed_nodes": blamed,
            "batch_size": len(fails), "lessons_stored": stored,
            "lessons": [{"node": l.node, "lesson": l.lesson[:120]} for l in lessons]}
