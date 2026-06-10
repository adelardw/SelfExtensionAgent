"""
GAIA — held-out бенчмарк (validation Level 1, текстовые задачи).

Честная проверка ГЕНЕРАЛИЗАЦИИ: задачи агент НЕ видел и под них не тюнили (в отличие
от наших 22 daily-сценариев). Scoring — GAIA-style exact match с нормализацией
чисел/строк/списков. GAIA сложен (топ-агенты на Level 1 ~40-50%, GPT-4+плагины ~15%).

Запуск: HF_TOKEN в .env, потом:
  .venv/bin/python -m src.eval.gaia_runner [N]   # N задач (по умолчанию 8)
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import time

os.environ.setdefault("AGENT_DRY_RUN", "1")
os.environ.setdefault("AGENT_SYSCALL_SANDBOX", "0")

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

SCENARIO_TIMEOUT = 240


# ── GAIA-style scorer (нормализация как в официальном question_scorer) ──

def _norm_number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def _norm_str(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def gaia_score(pred: str, gold: str) -> bool:
    """True, если ответ совпадает с золотым по GAIA-нормализации."""
    pred, gold = (pred or "").strip(), (gold or "").strip()
    # числа
    gn, pn = _norm_number(gold), None
    if gn is not None:
        # вытаскиваем последнее число из ответа агента (часто ответ в конце)
        nums = re.findall(r"-?[\d,]+\.?\d*", pred.replace("$", ""))
        for n in reversed(nums):
            pn = _norm_number(n)
            if pn is not None and pn == gn:
                return True
        return False
    # списки (через запятую/точку с запятой)
    if any(sep in gold for sep in [",", ";"]):
        gold_items = [_norm_str(x) for x in re.split(r"[,;]", gold) if x.strip()]
        pred_norm = _norm_str(pred)
        return all(gi in pred_norm for gi in gold_items)
    # строка: точное совпадение нормализованных ИЛИ вхождение золотого в ответ
    gn_s, pn_s = _norm_str(gold), _norm_str(pred)
    return gn_s == pn_s or (len(gn_s) > 2 and gn_s in pn_s)


# GAIA-протокол: агент завершает ответ строкой FINAL ANSWER → честный exact-match
# (иначе scorer ловит случайные числа в рассуждении = ложные срабатывания).
GAIA_PROTOCOL = (
    "\n\nВАЖНО: после рассуждения дай ПОСЛЕДНЕЙ строкой ровно:\n"
    "FINAL ANSWER: <краткий точный ответ — число БЕЗ единиц и запятых, либо слово/фраза, "
    "либо список через запятую>. Без пояснений в этой строке."
)


def _extract_final(answer: str) -> str:
    """Достаёт ответ после 'FINAL ANSWER:' (или весь текст, если протокол не соблюдён)."""
    m = re.search(r"FINAL ANSWER:\s*(.+?)\s*$", answer or "", re.I | re.M)
    return m.group(1).strip() if m else (answer or "")


def _load_tasks(n: int, with_files: bool = True) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    tok = os.getenv("HF_TOKEN")
    p = hf_hub_download("gaia-benchmark/GAIA", "2023/validation/metadata.level1.parquet",
                        repo_type="dataset", token=tok)
    df = pd.read_parquet(p)
    qcol = "Question" if "Question" in df.columns else next(c for c in df.columns if "question" in c.lower())
    acol = "Final answer" if "Final answer" in df.columns else next(c for c in df.columns if "answer" in c.lower())
    fcol = "file_name" if "file_name" in df.columns else None
    rows = []
    for _, r in df.iterrows():
        fname = str(r.get(fcol) or "").strip() if fcol else ""
        if fname and not with_files:
            continue
        rows.append({"q": str(r[qcol]), "a": str(r[acol]), "file": fname})
    return rows[:n]


def _attach_file(task: dict) -> str:
    """Скачивает файл-вложение задачи и читает его → добавляет к вопросу."""
    if not task.get("file"):
        return task["q"]
    try:
        from huggingface_hub import hf_hub_download

        from src.media import read_file
        p = hf_hub_download("gaia-benchmark/GAIA", f"2023/validation/{task['file']}",
                            repo_type="dataset", token=os.getenv("HF_TOKEN"))
        content = read_file(p, 10000)
        return f"{task['q']}\n\n=== ПРИЛОЖЕННЫЙ ФАЙЛ ({task['file']}) ===\n{content}"
    except Exception as e:  # noqa: BLE001
        return f"{task['q']}\n\n(файл {task['file']} не удалось прочитать: {e})"


async def run(n: int = 8) -> None:
    from src.agent import build_graph
    from src.usage import TokenTracker, cost_of

    tasks = _load_tasks(n)
    graph = build_graph()
    nfiles = sum(1 for t in tasks if t.get("file"))
    print(f"\n{'='*100}\nGAIA validation Level 1 (held-out, {len(tasks)} задач, из них с файлами: {nfiles})\n{'='*100}")
    correct, tot_cost = 0, 0.0
    for i, t in enumerate(tasks, 1):
        query = _attach_file(t) + GAIA_PROTOCOL  # файл (если есть) + протокол FINAL ANSWER
        tr = TokenTracker()
        t0 = time.monotonic()
        try:
            r = await asyncio.wait_for(
                graph.ainvoke({"query": query, "user_id": f"gaia_{i}", "chat_history": []},
                              config={"recursion_limit": 50, "callbacks": [tr]}),
                timeout=SCENARIO_TIMEOUT,
            )
            ans = r.get("final_answer", "") or ""
            mode = r.get("mode", "?")
        except Exception as e:  # noqa: BLE001
            ans, mode = f"[err {type(e).__name__}]", "error"
        final = _extract_final(ans)
        ok = gaia_score(final, t["a"])
        correct += ok
        cost = cost_of(tr.input, tr.output)
        tot_cost += cost
        tag = "📎" if t.get("file") else "  "
        print(f"\n[{i}]{tag}{'✅' if ok else '❌'} mode={mode} | gold={t['a']!r} | финал={final[:40]!r} | {round(time.monotonic()-t0)}с | ~${cost:.4f}")
        print(f"    Q: {t['q'][:110]}")
    print(f"\n{'='*100}")
    print(f"GAIA Level 1 accuracy: {correct}/{len(tasks)} = {correct/len(tasks):.0%}  ·  стоимость ${tot_cost:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    asyncio.run(run(n))
