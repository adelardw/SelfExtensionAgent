"""
Доказательство амортизации: ОДИН И ТОТ ЖЕ список задач прогоняется дважды на одном user_id.

Тезис паттерна («амортизированный агент»): ReAct / plan-execute / multi-agent имеют
~ПОСТОЯННУЮ стоимость задачи; здесь каждый успешный прогон оставляет артефакт (рецепт →
few-shot → привычка → навык), делающий похожие задачи ДЕШЕВЛЕ. Проверка: pass-2 должен
быть дешевле/быстрее pass-1 при той же (или лучшей) уверенности валидатора.

Запуск (ПЛАТНЫЙ — только по явному слову, ~десятки центов):
    uv run python scripts/amortize_bench.py            # дефолтные 4 задачи × 2 прохода
    uv run python scripts/amortize_bench.py tasks.txt  # свои задачи (по строке на задачу)
"""
import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Семейства задач НАМЕРЕННО разные (прогон №1 вскрыл: лексически похожие, но разные задачи
# ловили ложный «переспрос слабого» и дорожали). Амортизация меряется ПАРНО: та же задача
# cold vs warm — дисперсию между задачами это выносит за скобки.
DEFAULT_TASKS = [
    "Сделай саммари файла SETUP.md из текущей папки в 3 пунктах",
    "Посчитай: если класть 50000 в месяц под 12% годовых с капитализацией, сколько будет через 3 года?",
    "Создай файл /tmp/amortize_demo.md со списком 5 идей подарков коллеге-разработчику",
    "Создай файл /tmp/amortize_demo2.md с чек-листом из 5 пунктов для код-ревью",
]


async def run_pass(graph, tasks: list[str], user_id: str, label: str) -> dict:
    from src.usage import TokenTracker, cost_of

    total = {"in": 0, "out": 0, "sec": 0.0, "conf": [], "per_task": []}
    for i, q in enumerate(tasks, 1):
        tracker = TokenTracker()
        t0 = time.time()
        # Транзиентный сбой одной задачи не должен убивать ЭКСПЕРИМЕНТ (прогон №2 погиб
        # целиком на сетевой ошибке): помечаем задачу failed и продолжаем.
        try:
            res = await graph.ainvoke(
                {"query": q, "user_id": user_id, "session_id": f"{user_id}-s",
                 "chat_history": [{"role": "user", "content": q}]},
                config={"configurable": {"thread_id": str(uuid.uuid4())},
                        "recursion_limit": 50, "callbacks": [tracker]},
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [{label} {i}/{len(tasks)}] СБОЙ: {type(e).__name__}: {e}")
            res = {"confidence": 0.0}
        dt = time.time() - t0
        conf = res.get("confidence", 0.0)
        total["in"] += tracker.input; total["out"] += tracker.output
        total["sec"] += dt; total["conf"].append(conf)
        total["per_task"].append({"tok": tracker.total, "sec": dt, "conf": conf})
        print(f"  [{label} {i}/{len(tasks)}] {dt:5.1f}с  {tracker.total:>7} tok  "
              f"conf={conf:.0%}  {q[:60]}")
    total["usd"] = cost_of(total["in"], total["out"])
    return total


async def main() -> None:
    from src.agent import build_graph

    tasks = DEFAULT_TASKS
    if len(sys.argv) > 1:
        tasks = [l.strip() for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    graph = build_graph()
    uid = f"amortize-{uuid.uuid4().hex[:6]}"  # чистый юзер: pass-1 холодный по-настоящему
    print(f"== PASS 1 (холодный, user={uid}) ==")
    p1 = await run_pass(graph, tasks, uid, "cold")
    print(f"== PASS 2 (тёплый: рецепты/few-shots/прайоры того же юзера) ==")
    p2 = await run_pass(graph, tasks, uid, "warm")

    def fmt(p):
        avg_conf = sum(p["conf"]) / max(1, len(p["conf"]))
        return f"{p['in']+p['out']:>8} tok  ${p['usd']:.4f}  {p['sec']:6.1f}с  conf={avg_conf:.0%}"
    print("\n──── Амортизация (парно: та же задача cold → warm) ────")
    for i, (a, b, q) in enumerate(zip(p1["per_task"], p2["per_task"], tasks), 1):
        dtok = (1 - b["tok"] / max(1, a["tok"])) * 100
        print(f"  {i}. tok {a['tok']:>6}→{b['tok']:>6} ({dtok:+.0f}%)  "
              f"{a['sec']:5.1f}с→{b['sec']:5.1f}с  conf {a['conf']:.0%}→{b['conf']:.0%}  {q[:45]}")
    print(f"  pass-1 (cold): {fmt(p1)}")
    print(f"  pass-2 (warm): {fmt(p2)}")
    dt = (1 - (p2["in"] + p2["out"]) / max(1, p1["in"] + p1["out"])) * 100
    ds = (1 - p2["sec"] / max(0.1, p1["sec"])) * 100
    print(f"  Δ суммарно: токены {dt:+.0f}%  время {ds:+.0f}%  "
          f"(тезис подтверждён, если экономия >0 при не худшей conf)")


if __name__ == "__main__":
    asyncio.run(main())
