"""Живой бенч «симуляция пользователей»: LLM-персоны (src/eval/user_sim.py) ведут многоходовые
диалоги с РЕАЛЬНЫМ графом агента и дают уникальный UX-фидбек; судья оценивает каждый диалог
против цели персоны. Изоляция как у bench_scenarios: eval-режим (без вкладок/побочек, глобальные
сторы не пачкаются), временная память, DRY_RUN.

Запуск: uv run python bench_sim_users.py [persona_id ...]   (без аргументов — все персоны)
Отчёт: инкрементально в /tmp/sim_users_report.jsonl + сводка в stdout.
"""
import asyncio
import json
import os
import sys
import time
import uuid

os.environ.setdefault("AGENT_EVAL_MODE", "1")                    # без вкладок; глобальные сторы целы
os.environ.setdefault("AGENT_DRY_RUN", "1")                      # не дёргать реальное устройство
os.environ.setdefault("AGENT_MEMORY_DB", "/tmp/sim_users_memory.db")  # личная память цела

_OUT = "/tmp/sim_users_report.jsonl"
_TURN_TIMEOUT = 180


async def main() -> None:
    from main import make_checkpointer
    from src.graph.agent import build_graph
    from src.llm.llm import chat
    from src.runtime import hitl
    from src.eval.user_sim import (PERSONAS, DialogueVerdict, PersonaTurn, aggregate,
                                   format_transcript, persona_system_prompt, run_dialogue)

    hitl.set_work_mode("auto")
    wanted = set(sys.argv[1:])
    personas = [p for p in PERSONAS if not wanted or p["id"] in wanted]
    if not personas:
        print(f"нет таких персон; доступны: {', '.join(p['id'] for p in PERSONAS)}")
        return

    persona_llm = chat("fast", 0.7).with_structured_output(PersonaTurn)   # живость персоны
    judge_llm = chat("fast", 0).with_structured_output(DialogueVerdict)

    async def persona_step(p, transcript):
        return await persona_llm.ainvoke([
            ("system", persona_system_prompt(p)),
            ("human", f"Диалог на данный момент:\n\n{transcript}\n\n"
                      "Оцени последний ответ ассистента и сделай свой следующий ход."),
        ])

    async def judge(p, transcript):
        s = p["scenario"]
        return await judge_llm.ainvoke([
            ("system", "Ты — строгий судья качества диалога AI-ассистента с пользователем. "
                       "Судишь ФАКТЫ транскрипта, не намерения."),
            ("human", f"Персона: {p['name']} ({p['profile']})\n"
                      f"Цель: {s['goal']}\nКритерий успеха: {s['success']}\n\n"
                      f"Транскрипт:\n{transcript}\n\nВынеси вердикт."),
        ])

    open(_OUT, "w").close()
    async with make_checkpointer() as cp:
        graph = build_graph(cp)

        async def agent_call(query, chat_history, thread_id):
            r = await asyncio.wait_for(graph.ainvoke(
                {"query": query, "user_id": f"sim_{thread_id[:8]}", "session_id": thread_id,
                 "chat_history": chat_history},
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 50}),
                timeout=_TURN_TIMEOUT)
            return r.get("final_answer", "") or "(пустой ответ)"

        results = []
        for p in personas:
            sid = str(uuid.uuid4())
            t0 = time.time()
            print(f"\n=== {p['name']} ({p['id']}) ===", flush=True)
            try:
                res = await run_dialogue(agent_call, persona_step, p, sid)
                verdict = await judge(p, format_transcript(res["history"]))
                res["verdict"] = verdict.model_dump()
            except Exception as e:  # noqa: BLE001
                res = {"persona": p["id"], "history": [], "turns": 0,
                       "satisfaction": [], "feedbacks": [],
                       "error": f"{type(e).__name__}: {e}"[:200]}
            res["secs"] = round(time.time() - t0, 1)
            results.append(res)
            with open(_OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            v = res.get("verdict") or {}
            print(f"  ходов: {res['turns']}  satisfaction: {res.get('satisfaction')}  "
                  f"цель: {'✅' if v.get('goal_achieved') else '❌'}  "
                  f"grounded: {'✅' if v.get('grounded') else '❌'}  {res['secs']}s")
            for fb in res.get("feedbacks", []):
                print(f"  💬 {fb[:150]}")
            for i in (v.get("ux_issues") or []):
                print(f"  ⚠ {i[:150]}")
            if v.get("highlight"):
                print(f"  ⭐ {v['highlight'][:150]}")

    agg = aggregate(results)
    print("\n" + "=" * 72)
    print(f"ИТОГО: диалогов {agg['n']} · цель достигнута {agg['achieved']}/{agg['n']} · "
          f"grounded {agg['grounded']}/{agg['n']} · satisfaction {agg['avg_satisfaction']}/5 · "
          f"глубина {agg['avg_depth']}/5")
    if agg["issues"]:
        print("Проблемы UX по всем диалогам:")
        for i in agg["issues"]:
            print(f"  ⚠ {i[:160]}")
    print(f"Полный отчёт: {_OUT}")


if __name__ == "__main__":
    asyncio.run(main())
