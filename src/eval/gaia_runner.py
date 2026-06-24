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
import json
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

SCENARIO_TIMEOUT = 600 if os.getenv("AGENT_UNLEASH") == "1" else 240


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


def _wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson доверительный интервал для доли k/n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _load_tasks(n: int, levels: tuple = (1, 2, 3), offset: int = 0) -> list[dict]:
    """Грузит ВСЕ уровни validation (репрезентативно). n<=0 → весь набор."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    tok = os.getenv("HF_TOKEN")
    per_level: dict[int, list[dict]] = {}
    for lvl in levels:
        p = hf_hub_download("gaia-benchmark/GAIA", f"2023/validation/metadata.level{lvl}.parquet",
                            repo_type="dataset", token=tok)
        df = pd.read_parquet(p)
        qcol = "Question" if "Question" in df.columns else next(c for c in df.columns if "question" in c.lower())
        acol = "Final answer" if "Final answer" in df.columns else next(c for c in df.columns if "answer" in c.lower())
        fcol = "file_name" if "file_name" in df.columns else None
        per_level[lvl] = [{"q": str(r[qcol]), "a": str(r[acol]),
                           "file": str(r.get(fcol) or "").strip() if fcol else "", "level": lvl}
                          for _, r in df.iterrows()]
    # СТАБИЛЬНЫЙ глобальный порядок (round-robin по уровням) — один и тот же при любых
    # offset/limit, чтобы отказоустойчивый прогон по чанкам резюмировался детерминированно.
    full, i = [], 0
    while any(i < len(per_level[lvl]) for lvl in levels):
        for lvl in levels:
            if i < len(per_level[lvl]):
                full.append(per_level[lvl][i])
        i += 1
    sliced = full[offset:] if n <= 0 else full[offset:offset + n]
    return sliced


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
        hint = ""
        # PDF может содержать ФИГУРУ, где и лежит ответ (текст её не видит) → даём агенту путь
        # и подсказку прочитать фигуры через vision-тул read_pdf_figures.
        if str(task["file"]).lower().endswith(".pdf"):
            hint = (f"\n\n(Файл доступен по пути: {p} — если ответ в ФИГУРЕ/графике/диаграмме, "
                    f"а не в тексте выше, прочитай его инструментом read_pdf_figures.)")
        return f"{task['q']}\n\n=== ПРИЛОЖЕННЫЙ ФАЙЛ ({task['file']}) ===\n{content}{hint}"
    except Exception as e:  # noqa: BLE001
        return f"{task['q']}\n\n(файл {task['file']} не удалось прочитать: {e})"


async def run(n: int = 8, offset: int = 0, jsonl: str | None = None) -> None:
    """Прогон n задач с глобальным offset. Если задан jsonl — каждый результат пишется
    строкой JSON СРАЗУ (инкрементально, flush) → отказоустойчивый драйвер резюмирует после
    нативного краша (SIGABRT в либе) с точки, где остановились (см. scripts/gaia_resilient.py)."""
    # ЭКСПЕРИМЕНТ: тот же флаг, что и в main.py — GAIA можно гонять на composer-графе
    # (мета-контроллер примитивов) для сравнения с режимным графом.
    from src.graph.agent import config as _cfg
    if _cfg.get("experimental", {}).get("composer") or os.getenv("AGENT_EXPERIMENTAL"):
        from src.graph.agent_experimental import build_graph
        print("⚗  GAIA на experimental.composer (мета-контроллер примитивов)")
    else:
        from src.graph.agent import build_graph
    from src.llm.usage import TokenTracker, cost_of

    tasks = _load_tasks(n, offset=offset)
    graph = build_graph()
    nfiles = sum(1 for t in tasks if t.get("file"))
    print(f"\n{'='*100}\nGAIA validation (offset={offset}, {len(tasks)} задач, с файлами: {nfiles})\n{'='*100}")
    correct, tot_cost = 0, 0.0
    by_lvl: dict = {1: [0, 0], 2: [0, 0], 3: [0, 0]}  # level → [correct, total]
    jf = open(jsonl, "a", encoding="utf-8") if jsonl else None
    try:
        for j, t in enumerate(tasks):
            gidx = offset + j + 1  # глобальный 1-based индекс задачи
            query = _attach_file(t) + GAIA_PROTOCOL  # файл (если есть) + протокол FINAL ANSWER
            tr = TokenTracker()
            t0 = time.monotonic()
            try:
                # БЕЗ внешнего таймаута: граф терминируется сам (wall-бюджет шагов + SDK-таймаут
                # клиента). Гильотина рубила задачу за ~30с до естественного финиша и теряла её пустой
                # (11× TimeoutError на n=100). Пусть доходит до конца и отдаёт реальный ответ.
                r = await graph.ainvoke(
                    {"query": query, "user_id": f"gaia_{gidx}", "chat_history": []},
                    config={"recursion_limit": 50, "callbacks": [tr]})
                ans = r.get("final_answer", "") or ""
                mode = r.get("mode", "?")
            except Exception as e:  # noqa: BLE001
                ans, mode = f"[err {type(e).__name__}]", "error"
            secs = time.monotonic() - t0
            final = _extract_final(ans)
            ok = gaia_score(final, t["a"])
            correct += ok
            lvl = t.get("level", 1)
            by_lvl[lvl][0] += ok
            by_lvl[lvl][1] += 1
            cost = cost_of(tr.input, tr.output)
            tot_cost += cost
            tag = "📎" if t.get("file") else "  "
            print(f"[{gidx}]{tag}L{lvl}{'✅' if ok else '❌'} gold={t['a'][:25]!r} финал={final[:25]!r} {round(secs)}с ${cost:.4f}",
                  flush=True)
            if jf:  # ИНКРЕМЕНТАЛЬНО (до перехода к след. задаче) — переживёт нативный краш
                # sec/in/out/cached — время ответа и токены (цель: сохранить тайминги и прочее);
                # cached = входные токены из авто-prefix-cache (K3-телеметрия).
                jf.write(json.dumps({"idx": gidx, "level": lvl, "ok": bool(ok), "gold": t["a"],
                                     "final": final[:200], "mode": mode, "cost": cost,
                                     "sec": round(secs, 1), "in": tr.input, "out": tr.output,
                                     "cached": tr.cached}, ensure_ascii=False) + "\n")
                jf.flush()
    finally:
        if jf:
            jf.close()
    n_tot = len(tasks)
    if n_tot:
        lo, hi = _wilson(correct, n_tot)
        print(f"\n{'='*100}")
        print(f"GAIA (offset={offset}, {n_tot} задач): {correct}/{n_tot} = {correct/n_tot:.1%}  "
              f"[95% Wilson CI: {lo:.1%}–{hi:.1%}]  ·  стоимость ${tot_cost:.4f}")
        for lvl in (1, 2, 3):
            c, tnum = by_lvl[lvl]
            if tnum:
                print(f"  Level {lvl}: {c}/{tnum} = {c/tnum:.0%}")
        print("=" * 100)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=0)  # 0 = весь validation (165)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--jsonl", type=str, default=None)
    a = ap.parse_args()
    asyncio.run(run(a.n, offset=a.offset, jsonl=a.jsonl))
