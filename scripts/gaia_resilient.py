"""
Отказоустойчивый GAIA-прогон: нативный краш в либе (SIGABRT, leaked semaphore) не должен
убивать весь замер. Каждый «чанк» оставшихся задач исполняется в ОТДЕЛЬНОМ подпроцессе с
инкрементальным JSONL; при крахе драйвер резюмирует с точки остановки, а задачу, на которой
подпроцесс упал не записав результат, помечает как crash-err и пропускает.

Запуск (env пробрасывается — AGENT_MEMORY_DB / AGENT_EVAL_MODE / AGENT_NO_BROWSER и пр.):
  AGENT_MEMORY_DB=/tmp/g.db AGENT_EVAL_MODE=1 AGENT_NO_BROWSER=1 \
  .venv/bin/python scripts/gaia_resilient.py 100 --jsonl data/eval/gaia100.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _count(jsonl: Path) -> int:
    if not jsonl.exists():
        return 0
    return sum(1 for _ in jsonl.open(encoding="utf-8"))


def _aggregate(jsonl: Path) -> None:
    rows = [json.loads(l) for l in jsonl.open(encoding="utf-8") if l.strip()]
    n = len(rows)
    if not n:
        print("нет результатов")
        return
    correct = sum(1 for r in rows if r.get("ok"))
    by: dict[int, list[int]] = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    crashes = sum(1 for r in rows if r.get("mode") == "crash")
    cost = sum(float(r.get("cost", 0.0)) for r in rows)
    for r in rows:
        lvl = int(r.get("level", 1))
        by.setdefault(lvl, [0, 0])
        by[lvl][0] += bool(r.get("ok"))
        by[lvl][1] += 1
    # Тайминги ответов (цель: сохранить время и прочее). sec пишет gaia_runner per-task.
    secs = sorted(float(r["sec"]) for r in rows if r.get("sec") is not None)
    avg_s = sum(secs) / len(secs) if secs else 0.0
    med_s = secs[len(secs) // 2] if secs else 0.0
    cached = sum(int(r.get("cached", 0) or 0) for r in rows)
    toks_in = sum(int(r.get("in", 0) or 0) for r in rows)
    hit = (cached / toks_in) if toks_in else 0.0
    print("\n" + "=" * 100)
    print(f"GAIA РЕЗИЛЬЕНТНЫЙ ИТОГ: {correct}/{n} = {correct / n:.1%}  ·  крашей пропущено: {crashes}  ·  стоимость ${cost:.4f}")
    for lvl in (1, 2, 3):
        c, t = by.get(lvl, [0, 0])
        if t:
            print(f"  Level {lvl}: {c}/{t} = {c / t:.0%}")
    if secs:
        print(f"  Время ответа: сред {avg_s:.0f}с · медиана {med_s:.0f}с · макс {secs[-1]:.0f}с"
              f"  ·  prefix-cache hit: {hit:.0%}")
    print("=" * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()
    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    done = _count(jsonl)
    while done < args.n:
        print(f"\n>>> чанк с offset={done} (осталось {args.n - done}) <<<", flush=True)
        rc = subprocess.run(
            [sys.executable, "-m", "src.eval.gaia_runner", str(args.n - done),
             "--offset", str(done), "--jsonl", str(jsonl)],
            env=os.environ.copy(),
        )
        new = _count(jsonl)
        if new == done:
            # подпроцесс упал, НЕ записав результат текущей задачи → пометить crash-err и пропустить
            with jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"idx": done + 1, "level": 1, "ok": False, "gold": "",
                                    "final": f"[crash rc={rc.returncode}]", "mode": "crash", "cost": 0.0},
                                   ensure_ascii=False) + "\n")
            print(f"!!! подпроцесс упал (rc={rc.returncode}) на задаче {done + 1} — пропускаю", flush=True)
            new = done + 1
        done = new

    _aggregate(jsonl)


if __name__ == "__main__":
    main()
