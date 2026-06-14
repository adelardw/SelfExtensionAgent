"""Сводка по GAIA-прогону из jsonl-пруфа: accuracy + Wilson 95% CI + разбивка по уровням.

    python scripts/gaia_summary.py eval_results/gaia/gaia100_strong_tier.jsonl
    python scripts/gaia_summary.py eval_results/gaia/*.jsonl

Каждая строка jsonl — одна задача: {idx, level, mode, ok, gold, final, cost}.
Скрипт не делает сетевых вызовов — только пересчитывает метрики из готового файла.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import sqrt


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(path: str) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(rows)
    ok = sum(1 for r in rows if r.get("ok"))
    cost = sum((r.get("cost") or 0) for r in rows)
    errs = sum(1 for r in rows
               if isinstance(r.get("final"), str) and "error" in r["final"].lower()[:12])
    lo, hi = wilson(ok, n)
    print(f"\n=== {path} ===")
    print(f"  задач={n}  верно={ok}  errored={errs}  flat-cost(usage.py)=${cost:.2f}")
    print(f"  accuracy = {ok/n:.1%}  Wilson95% = [{lo:.1%}, {hi:.1%}]")
    lv = defaultdict(lambda: [0, 0])
    for r in rows:
        lv[r.get("level")][0] += 1
        lv[r.get("level")][1] += 1 if r.get("ok") else 0
    for k in sorted(lv, key=lambda x: str(x)):
        tot, good = lv[k]
        l2, h2 = wilson(good, tot)
        print(f"    L{k}: {good}/{tot} ({good/tot:.0%})  Wilson95% [{l2:.0%}, {h2:.0%}]")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("использование: python scripts/gaia_summary.py <файл.jsonl> [...]")
        raise SystemExit(2)
    for p in args:
        summarize(p)
