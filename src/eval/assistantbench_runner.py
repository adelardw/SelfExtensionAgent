"""
AssistantBench — held-out бенч РЕАЛЬНЫХ повседневных веб-задач (студии рядом, хайки,
вероятность дождя…). Сложнее GAIA для локального поиска: потолок <25% даже у топ-веб-агентов.

Scoring — приближение официального (частичный кредит): числа с толерансом, списки F1
по пунктам (recall), строки по вхождению. Запуск:
  .venv/bin/python -m src.eval.assistantbench_runner [N] [difficulty]
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import time

os.environ.setdefault("AGENT_DRY_RUN", "1")
os.environ.setdefault("AGENT_SYSCALL_SANDBOX", "0")
os.environ.setdefault("AGENT_EVAL_MODE", "1")  # не загрязнять глобальные few-shots бенч-запросами

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

SCENARIO_TIMEOUT = 240
PROTOCOL = ("\n\nВ конце дай строку 'FINAL ANSWER: <ответ>' — для списка перечисли пункты "
            "через перенос строки или запятую; для числа — только число.")


def _num(s: str):
    try:
        return float(str(s).replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def ab_score(pred: str, gold: str) -> float:
    """Частичный кредит [0..1] по типу ответа (приближение метрики AssistantBench)."""
    pred = re.sub(r".*FINAL ANSWER:\s*", "", pred or "", flags=re.I | re.S).strip()
    gold_items = [g.strip() for g in str(gold).split("\n") if g.strip()]
    if not gold_items:
        return 0.0
    # число
    if len(gold_items) == 1 and _num(gold_items[0]) is not None:
        gn = _num(gold_items[0])
        for cand in re.findall(r"-?[\d,]+\.?\d*", pred):
            pn = _num(cand)
            if pn is not None and (abs(pn - gn) <= max(0.05 * abs(gn), 0.5)):
                return 1.0
        return 0.0
    # список → доля найденных пунктов (recall)
    if len(gold_items) > 1:
        pn = _norm(pred)
        hits = sum(1 for g in gold_items if _norm(g) and _norm(g) in pn)
        return round(hits / len(gold_items), 3)
    # строка → вхождение
    g = _norm(gold_items[0])
    return 1.0 if g and g in _norm(pred) else 0.0


def _load(n: int, difficulty: str = "") -> list[dict]:
    import json

    from huggingface_hub import hf_hub_download
    p = hf_hub_download("AssistantBench/AssistantBench", "assistant_bench_v1.0_dev.jsonl",
                        repo_type="dataset", token=os.getenv("HF_TOKEN"))
    rows = [json.loads(l) for l in open(p) if l.strip()]
    rows = [r for r in rows if r.get("answer")]
    if difficulty:
        rows = [r for r in rows if str(r.get("difficulty", "")).lower() == difficulty.lower()]
    return rows[:n]


async def run(n: int = 6, difficulty: str = "") -> None:
    from src.agent import build_graph
    from src.usage import TokenTracker, cost_of

    tasks = _load(n, difficulty)
    graph = build_graph()
    print(f"\n{'='*100}\nAssistantBench dev (held-out, {len(tasks)} задач{', '+difficulty if difficulty else ''})\n{'='*100}")
    total, tot_cost = 0.0, 0.0
    for i, t in enumerate(tasks, 1):
        tr = TokenTracker()
        t0 = time.monotonic()
        try:
            r = await asyncio.wait_for(
                graph.ainvoke({"query": t["task"] + PROTOCOL, "user_id": f"ab_{i}", "chat_history": []},
                              config={"recursion_limit": 50, "callbacks": [tr]}),
                timeout=SCENARIO_TIMEOUT,
            )
            ans, mode = r.get("final_answer", "") or "", r.get("mode", "?")
        except Exception as e:  # noqa: BLE001
            ans, mode = f"[err {type(e).__name__}]", "error"
        sc = ab_score(ans, t["answer"])
        total += sc
        cost = cost_of(tr.input, tr.output)
        tot_cost += cost
        mark = "✅" if sc >= 0.999 else ("🟡" if sc > 0 else "❌")
        print(f"\n[{i}] {mark} score={sc:.2f} | diff={t.get('difficulty')} | {round(time.monotonic()-t0)}с | ~${cost:.4f}")
        print(f"    Q: {t['task'][:110]}")
        print(f"    gold: {t['answer'][:70].replace(chr(10),' | ')}")
        print(f"    A: {ans[:120].strip().replace(chr(10),' ')}")
    print(f"\n{'='*100}")
    print(f"AssistantBench accuracy (частичный): {total/len(tasks):.0%}  ·  стоимость ${tot_cost:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    diff = sys.argv[2] if len(sys.argv) > 2 else ""
    asyncio.run(run(n, diff))
