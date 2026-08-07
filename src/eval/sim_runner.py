"""Ядро тестового стенда: ОДИН ход диалога с живым графом + ТЕЛЕМЕТРИЯ хода.

Зачем: раньше стенд отдавал судьям только текст ответа, и они ГАДАЛИ по логам («считал ли
инструментом?», «работал ли поиск?», «сколько заняло?») — в вердиктах так и писали
«used_compute_tool: неясно», «latency: не измерялось». Теперь каждый ход возвращает
структурный `meta`: режим, длительность, статистика поиска/чтений, вызванные инструменты,
сработавшие гейты, артефакты. Судьи судят по фактам, а прогоны становятся аггрегируемыми.

Используется и CLI-драйвером (scripts/sim_chat_driver.py), и регресс-набором
(bench_regressions.py) — единый путь вызова графа, без дублирования.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

HIST_DIR = Path(os.getenv("AGENT_SIM_HIST_DIR") or "/tmp/sim_chat_threads")
JOURNAL = Path(os.getenv("AGENT_SIM_JOURNAL") or "data/eval/harness_runs.jsonl")
DEFAULT_TIMEOUT_S = float(os.getenv("AGENT_SIM_TIMEOUT_S") or 900)

# Маркеры, которые ноды печатают в stdout — по ним видно, какие защиты сработали в ходе.
# Держим ЗДЕСЬ единым списком: судья не должен знать внутренние строки агента.
MARKER_PATTERNS: dict[str, str] = {
    "antifab": r"\[AntiFab\]",                       # гейт анти-фабрикации подменил ответ
    "circuit_open": r"circuit OPEN",                  # поиск отключён после N провалов
    "searxng_empty": r"SearXNG жив, но 0 результатов",
    "search_fallback": r"иду в fallback",
    "skill_skipped": r"\[SkillManager\] Skipped",     # навык не загрузился (мёртв)
    "validation_skipped": r"consensus skipped|валидатор не распарсился",
    "recipe": r"\[Recipe\]",
    "distill": r"\[Distill\]",
    "clarify_gate": r"Чтобы сделать полезно, уточни",
}


def _count_markers(log: str) -> dict[str, int]:
    """Сколько раз каждый маркер встретился в служебном выводе хода."""
    return {k: len(re.findall(p, log)) for k, p in MARKER_PATTERNS.items()
            if re.search(p, log)}


def _tools_called(state: dict) -> list[str]:
    """Имена РЕАЛЬНО вызванных инструментов из сообщений прогона (заземление «сделал»)."""
    out: list[str] = []
    for m in (state.get("messages") or []):
        for tc in (getattr(m, "tool_calls", None) or []):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name:
                out.append(str(name))
    return out


class _Tee(io.TextIOBase):
    """stdout → и в консоль (судья видит живой лог), и в буфер (для телеметрии)."""

    def __init__(self, real):
        self._real = real
        self.buf = io.StringIO()

    def write(self, s: str) -> int:  # noqa: D102
        self.buf.write(s)
        return self._real.write(s)

    def flush(self) -> None:  # noqa: D102
        self._real.flush()


def _load_history(thread: str) -> list[dict]:
    f = HIST_DIR / f"{thread}.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    except Exception:  # noqa: BLE001 — битую историю не тащим в прогон
        return []


def _save_history(thread: str, history: list[dict]) -> None:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    (HIST_DIR / f"{thread}.json").write_text(
        json.dumps(history[-40:], ensure_ascii=False), encoding="utf-8")


def journal_append(record: dict) -> None:
    """Дописать ход в машиночитаемый журнал прогонов (для сравнения раундов/трендов)."""
    try:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with JOURNAL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — журнал не должен ронять прогон
        pass


async def run_turn(thread: str, message: str, timeout_s: float = DEFAULT_TIMEOUT_S,
                   graph=None, checkpointer=None) -> dict:
    """Один ход: реплика → {answer, meta, timed_out}. Историю сохраняет ДО invoke (ход не
    теряется при таймауте) и после. Телеметрию снимает ВНУТРИ request_scope — иначе cleanup
    по выходу уже вычистит счётчики прогона."""
    from src.runtime import hitl, run_context, runbudget

    hitl.set_work_mode("auto")
    history = _load_history(thread)
    history.append({"role": "user", "content": message})
    _save_history(thread, history)

    run_id = f"simchat_{thread}"
    tee = _Tee(__import__("sys").stdout)
    t0 = time.monotonic()
    state: dict = {}
    timed_out = False
    meta: dict = {}

    async def _invoke(g):
        nonlocal state, timed_out, meta
        with run_context.request_scope(run_id, f"sim_{thread}"):
            try:
                state = await asyncio.wait_for(g.ainvoke(
                    {"query": message, "user_id": f"sim_{thread}", "session_id": run_id,
                     "chat_history": history[-20:]},
                    config={"configurable": {"thread_id": run_id}, "recursion_limit": 50}),
                    timeout=timeout_s)
            except asyncio.TimeoutError:
                timed_out = True
            # снимаем ВНУТРИ scope: cleanup на выходе обнулит счётчики
            s_att, s_ok = run_context.search_stats()
            r_att, r_ok = run_context.page_read_stats()
            meta = {
                "search_attempts": s_att, "search_ok": s_ok,
                "page_reads": r_att, "page_reads_ok": r_ok,
                "tool_calls_noted": run_context.tool_calls_count(),
                # имена тулов из run_context: act/research не возвращают свои messages наружу,
                # поэтому только так видно, ЧЕМ агент реально работал
                "tools_called": run_context.tool_names(),
                "human_wait_s": round(runbudget.human_wait_seconds(), 1),
                "artifacts": [a.get("name", "") for a in run_context.artifacts()],
            }

    with redirect_stdout(tee):
        if graph is not None:
            await _invoke(graph)
        else:
            from main import make_checkpointer
            from src.graph.agent import build_graph

            async with make_checkpointer() as cp:
                await _invoke(build_graph(cp))

    elapsed = round(time.monotonic() - t0, 1)
    answer = (state.get("final_answer") or "").strip()
    if timed_out:
        answer = (f"(таймаут стенда: ход не уложился в {timeout_s:.0f}с. Реплика сохранена "
                  "в историю; повтори или подними AGENT_SIM_TIMEOUT_S.)")
    elif not answer:
        answer = "(пустой ответ)"
    else:
        history.append({"role": "assistant", "content": answer})
        _save_history(thread, history)

    log = tee.buf.getvalue()
    meta.update({
        "thread": thread, "elapsed_s": elapsed, "timed_out": timed_out,
        "mode": state.get("mode", ""), "route": state.get("route", ""),
        "skills": state.get("selected_skills", []) or [],
        "steps": state.get("steps_executed", 0),
        "confidence": round(float(state.get("confidence", 0) or 0), 2),
        "validation_passed": bool(state.get("validation_passed", False)),
        # к именам из run_context добавляем тулы из state.messages (deliberate-путь) — дедуп
        # с сохранением порядка; пусто → агент не вызывал инструментов вообще
        "tools_called": list(dict.fromkeys((meta.get("tools_called") or []) + _tools_called(state))),
        "markers": _count_markers(log),
        "answer_chars": len(answer),
        "answer_urls": len(re.findall(r"https?://", answer)),
    })
    journal_append({"ts": time.time(), "message": message[:200],
                    "answer": answer[:2000], **meta})
    return {"answer": answer, "meta": meta, "timed_out": timed_out}


def format_turn_output(res: dict) -> str:
    """Вывод для судьи: ответ + машиночитаемая телеметрия хода."""
    return (f"\n===ANSWER===\n{res['answer']}\n"
            f"\n===META===\n{json.dumps(res['meta'], ensure_ascii=False)}")


def reset_thread(thread: str) -> None:
    """Забыть историю треда (для чистого прогона сценария)."""
    f = HIST_DIR / f"{thread}.json"
    if f.exists():
        f.unlink()


def journal_read(limit: Optional[int] = None) -> list[dict]:
    """Прочитать журнал прогонов (для сравнения раундов)."""
    if not JOURNAL.exists():
        return []
    rows = []
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return rows[-limit:] if limit else rows
