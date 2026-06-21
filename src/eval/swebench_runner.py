"""
SWE-bench Lite харнесс (настоящий, признанный бенчмарк).

Режимы:
  --gold N   : взять N инстансов, прогнать swebench-оценку с ЭТАЛОННЫМ патчем
               (валидирует, что наш пайплайн оценки корректен → resolved=true).
  --agent N  : наш агент пытается решить (репо + проблема → git diff как предсказание),
               затем swebench-оценка. ЧЕСТНО: мы не кодинг-агент, ожидается низкий скор.

Оценка использует официальный пакет `swebench` + Docker. Отчёт парсится из report-json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Lite"
OUT = Path("data/eval")


def _load(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="test")
    return [ds[i] for i in range(min(n, len(ds)))]


def _write_preds(insts: list[dict], use_gold: bool, model_name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "swebench_preds.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ins in insts:
            patch = ins["patch"] if use_gold else _agent_patch(ins)
            f.write(json.dumps({
                "instance_id": ins["instance_id"],
                "model_name_or_path": model_name,
                "model_patch": patch,
            }) + "\n")
    return path


def _agent_patch(ins: dict) -> str:
    """
    Наш агент пытается решить инстанс: клон репо @base_commit → агент правит файлы →
    git diff. ЧЕСТНО: без кодинг-харнесса это слабо; возвращает пустой патч при неудаче.
    """
    import asyncio
    import uuid

    repo, commit = ins["repo"], ins["base_commit"]
    work = Path(tempfile.mkdtemp(prefix="swe_"))
    try:
        subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(work)],
                       capture_output=True, timeout=300)
        subprocess.run(["git", "checkout", commit], cwd=work, capture_output=True, timeout=60)

        import src.graph.agent as A
        from langgraph.checkpoint.memory import MemorySaver

        graph = A.build_graph(MemorySaver())
        task = (f"Репозиторий по пути {work}. Исправь проблему, редактируя файлы в этом пути:\n\n"
                f"{ins['problem_statement'][:2000]}\n\nИспользуй файловые инструменты, не объясняй — внеси правки.")
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 60}
        asyncio.run(graph.ainvoke({"query": task, "user_id": "swe", "chat_history": []}, config=cfg))

        diff = subprocess.run(["git", "diff"], cwd=work, capture_output=True, text=True, timeout=60)
        return diff.stdout
    except Exception as e:  # noqa: BLE001
        print(f"[swe] agent patch failed for {ins['instance_id']}: {e}")
        return ""


def _evaluate(preds: Path, run_id: str, instance_ids: list[str]) -> dict:
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", DATASET, "--predictions_path", str(preds),
           "--max_workers", "1", "--run_id", run_id, "--cache_level", "env",
           "--instance_ids", *instance_ids]
    print("[swe] eval:", " ".join(cmd))
    subprocess.run(cmd, timeout=3600)
    # отчёт: <model>.<run_id>.json в cwd
    reports = list(Path(".").glob(f"*.{run_id}.json"))
    if not reports:
        return {"error": "report not found"}
    return json.loads(reports[0].read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=int, default=0)
    ap.add_argument("--agent", type=int, default=0)
    args = ap.parse_args()
    use_gold = args.gold > 0
    n = args.gold or args.agent or 1
    model = "gold" if use_gold else "self-extension-agent"

    insts = _load(n)
    ids = [i["instance_id"] for i in insts]
    print(f"[swe] instances: {ids}")
    preds = _write_preds(insts, use_gold, model)
    report = _evaluate(preds, run_id=f"se_{'gold' if use_gold else 'agent'}", instance_ids=ids)
    print(json.dumps({k: report.get(k) for k in
                      ("total_instances", "submitted_instances", "completed_instances",
                       "resolved_instances", "unresolved_instances", "error_instances")},
                     indent=2))


if __name__ == "__main__":
    main()
