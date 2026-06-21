"""
Eval-харнесс: метрики качества агента + честная проверка, что backward улучшает.

Эксперимент: baseline на всём наборе → оптимизируем промпт по failures из TRAIN
(textual-gradient) → меряем прирост на ОТЛОЖЕННОМ TEST (генерализация, не подгонка).
Результаты + график сохраняются в data/eval/.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from src.llm.llm import chat
from ..memory import MemoryStore, build_embedder

TASKS = json.loads((Path(__file__).parent / "tasks.json").read_text(encoding="utf-8"))
OUT = Path("data/eval")
_MODEL_CFG = __import__("omegaconf").OmegaConf.load("config.yml").get("model", {}).get("name", "gpt-4o-mini")


class _Judge(BaseModel):
    met: list[bool] = Field(description="По каждому критерию: выполнен ли (true/false), в том же порядке")
    note: str = Field(description="Кратко что не так", default="")


_judge_llm = None


def _judge(task: dict, answer: str) -> tuple[float, str]:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = chat(_MODEL_CFG).with_structured_output(_Judge)
    crit = "\n".join(f"{i+1}. {c}" for i, c in enumerate(task["criteria"]))
    prompt = (f"Оцени ответ агента по критериям (строго).\n\nЗапрос: {task['query']}\n\n"
              f"Критерии:\n{crit}\n\nОтвет агента:\n{answer}\n\n"
              f"Для каждого критерия — выполнен ли он.")
    try:
        r = _judge_llm.invoke(prompt)
        met = r.met[: len(task["criteria"])]
        score = sum(1 for m in met if m) / max(1, len(task["criteria"]))
        return score, r.note
    except Exception as e:  # noqa: BLE001
        return 0.0, f"judge error: {e}"


async def run_suite(graph, tasks: list[dict]) -> dict:
    """Прогоняет задачи через граф, судит, возвращает {scores per task, by split}."""
    from langgraph.checkpoint.memory import MemorySaver  # noqa: F401

    results = {}
    for t in tasks:
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}
        try:
            r = await graph.ainvoke({"query": t["query"], "user_id": "eval", "chat_history": []}, config=cfg)
            ans = r.get("final_answer", "")
        except Exception as e:  # noqa: BLE001
            ans = f"(ошибка: {e})"
        score, note = _judge(t, ans)
        results[t["id"]] = {"split": t["split"], "score": score, "answer": ans[:200], "note": note}
    return results


def _avg(results: dict, split: str) -> float:
    vals = [v["score"] for v in results.values() if v["split"] == split]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _save(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _chart(baseline: dict, improved: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ["train", "test"]
    b = [_avg(baseline, s) for s in splits]
    im = [_avg(improved, s) for s in splits]
    x = range(len(splits))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - 0.2 for i in x], b, width=0.4, label="baseline", color="#888")
    ax.bar([i + 0.2 for i in x], im, width=0.4, label="после backward", color="#2a9d8f")
    ax.set_xticks(list(x)); ax.set_xticklabels(splits)
    ax.set_ylim(0, 1); ax.set_ylabel("rubric score"); ax.set_title("Self-improvement (backward) — до/после")
    ax.legend()
    for i, (vb, vi) in enumerate(zip(b, im)):
        ax.text(i - 0.2, vb + 0.02, f"{vb:.2f}", ha="center")
        ax.text(i + 0.2, vi + 0.02, f"{vi:.2f}", ha="center")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "improvement.png"
    fig.tight_layout(); fig.savefig(path, dpi=120)
    return str(path)


async def experiment() -> dict:
    """baseline → оптимизация промпта по TRAIN-failures → re-eval → дельта на TEST + график."""
    import src.graph.agent as A
    from src.improve import prompt_store, SelfLearningPipe

    # изоляция: временная память + временный ParamStore (не пачкаем боевые)
    A.memory_store = MemoryStore(tempfile.mktemp(suffix=".db"), embedder=build_embedder(False))
    prompt_store.PARAMS_FILE = Path(tempfile.mktemp(suffix=".json"))
    from langgraph.checkpoint.memory import MemorySaver

    graph = A.build_graph(MemorySaver())

    baseline = await run_suite(graph, TASKS)

    # failures из TRAIN (score < 1.0) → textual-gradient оптимизация промпта быстрого ответа
    failures = [
        {"query": t["query"], "answer": baseline[t["id"]]["answer"], "feedback": baseline[t["id"]]["note"]}
        for t in TASKS if t["split"] == "train" and baseline[t["id"]]["score"] < 1.0
    ]
    opt = {"status": "no_failures"}
    if failures:
        opt = SelfLearningPipe(A.memory_store).optimize_role("fast_answer", failures, accept=True)

    improved = await run_suite(graph, TASKS)  # re-eval (override уже активен, если принят)

    summary = {
        "baseline": {"train": _avg(baseline, "train"), "test": _avg(baseline, "test")},
        "improved": {"train": _avg(improved, "train"), "test": _avg(improved, "test")},
        "delta_test": round(_avg(improved, "test") - _avg(baseline, "test"), 3),
        "optimization": opt,
        "per_task_baseline": baseline,
        "per_task_improved": improved,
    }
    _save("experiment.json", summary)
    try:
        summary["chart"] = _chart(baseline, improved)
    except Exception as e:  # noqa: BLE001
        summary["chart"] = f"chart failed: {e}"
    return summary


if __name__ == "__main__":
    res = asyncio.run(experiment())
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_task_baseline", "per_task_improved")},
                     ensure_ascii=False, indent=2))
