"""Драйвер многоходового диалога с агентом для ВНЕШНИХ валидаторов (Claude-сабагенты и пр.):
один вызов = одна реплика пользователя → печатает ответ агента И телеметрию хода.

Использование:
    uv run python scripts/sim_chat_driver.py <thread_name> "реплика пользователя"
    uv run python scripts/sim_chat_driver.py --reset <thread_name>   # забыть тред

Вывод:
    ===ANSWER===   текст ответа агента (всё выше — служебный лог нод)
    ===META===     JSON: mode, elapsed_s, tools_called, search_*, markers, artifacts …

Изоляция как у бенчей: eval-режим (без вкладок/побочек, глобальные сторы целы), временная
память (общая на тред — контуры памяти работают), DRY_RUN. Логика прогона — в
src/eval/sim_runner.py (тот же путь использует bench_regressions.py).
"""
import asyncio
import os
import sys

os.environ.setdefault("AGENT_EVAL_MODE", "1")
os.environ.setdefault("AGENT_DRY_RUN", "1")
os.environ.setdefault("AGENT_MEMORY_DB", "/tmp/sim_chat_memory.db")
os.environ.setdefault("AGENT_CLARIFY_SHORTCIRCUIT", "1")  # стенд: юзер ответит следующим ходом


async def main() -> None:
    from src.eval.sim_runner import format_turn_output, reset_thread, run_turn

    args = sys.argv[1:]
    if args and args[0] == "--reset":
        if len(args) < 2:
            print("usage: sim_chat_driver.py --reset <thread_name>")
            raise SystemExit(2)
        reset_thread(args[1])
        print(f"тред '{args[1]}' очищен")
        return
    if len(args) < 2:
        print("usage: sim_chat_driver.py <thread_name> <message>")
        raise SystemExit(2)

    res = await run_turn(args[0], args[1])
    print(format_turn_output(res))
    raise SystemExit(3 if res["timed_out"] else 0)


if __name__ == "__main__":
    asyncio.run(main())
