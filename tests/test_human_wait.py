"""Вопрос человеку ЖДЁТ СКОЛЬКО УГОДНО (фишка HITL, как AskUserQuestion в Claude Code):
часы прогона на время раздумий СТОЯТ (runbudget.human_pause), дедлайн шага к ask_user и
HITL-обёрнутым тулам НЕ применяется. Offline."""
import asyncio
import time

import pytest

from src.runtime import run_context, runbudget


def test_human_pause_excludes_wait_from_elapsed():
    with run_context.request_scope("run_hp1", "u"):
        runbudget.reset()
        with runbudget.human_pause():
            time.sleep(0.25)                       # «человек думает»
        assert runbudget.human_wait_seconds() >= 0.2
        assert runbudget.elapsed() < 0.15          # работа агента ≈ 0, раздумья не в счёт


def test_human_pause_prevents_false_exhaustion():
    """Прогон с бюджетом 0.2с не считается исчерпанным, если 0.3с ждали человека."""
    with run_context.request_scope("run_hp2", "u"):
        runbudget.reset()
        with runbudget.human_pause():
            time.sleep(0.3)
        assert not runbudget.exhausted(token_limit=10**9, sec_limit=0.2)
        time.sleep(0.25)                           # а вот это уже РАБОТА агента
        assert runbudget.exhausted(token_limit=10**9, sec_limit=0.2)


def test_human_pause_reentrant_and_scoped():
    with run_context.request_scope("run_hp3", "u"):
        runbudget.reset()
        with runbudget.human_pause():
            time.sleep(0.1)
            with runbudget.human_pause():          # вложенный ask внутри HITL
                time.sleep(0.1)
        assert runbudget.human_wait_seconds() >= 0.2
    with run_context.request_scope("run_hp4", "u"):
        runbudget.reset()
        assert runbudget.human_wait_seconds() == 0.0  # чужой прогон не видит паузы


def test_clarify_ask_pauses_clock_and_waits_indefinitely():
    """clarify.ask с медленным каналом: ответ дожидается, часы прогона не тикают."""
    from src.interface import clarify

    async def _slow_clarifier(items):
        await asyncio.sleep(0.3)                   # человек думал дольше «дедлайна»
        return ["CRM-системы"]

    clarify.set_clarifier(_slow_clarifier)
    try:
        with run_context.request_scope("run_hp5", "u"):
            runbudget.reset()
            clarify.reset_ledger()
            res = asyncio.run(clarify.ask([{"question": "Что сравниваем?", "options": []}]))
            assert res[0]["answer"] == "CRM-системы" and res[0]["status"] == "answered"
            assert runbudget.human_wait_seconds() >= 0.25
            assert runbudget.elapsed() < 0.2       # раздумья не съели бюджет
    finally:
        clarify.set_clarifier(None)


def test_ask_user_tool_is_marked_human_wait():
    from src.graph.agent import _HUMAN_WAIT_TOOLS

    assert "ask_user" in _HUMAN_WAIT_TOOLS         # исполнитель не накладывает дедлайн шага


def test_ask_user_is_never_bounded_even_without_pause():
    """Уточнение НЕ ограничивается вообще (как AskUserQuestion): даже если канал забыл
    обернуться в human_pause, ask_user ждёт — дедлайн к нему не применяется в принципе."""
    import inspect

    from src.graph import agent as A

    src = inspect.getsource(A._exec_direct)
    i_ask = src.index("_HUMAN_WAIT_TOOLS")
    branch = src[i_ask:i_ask + 700]
    # в ветке ask_user — голый await, без wait_for/bounded-обёртки
    assert "await t.ainvoke" in branch
    head = branch[:branch.index("elif")]
    assert "wait_for" not in head and "_await_tool_work_bounded" not in head


def test_hitl_wrapped_tool_carries_guard_flag():
    """HITL-обёртка помечает тул структурно → шаг не режет подтверждение дедлайном."""
    from langchain_core.tools import StructuredTool

    from src.runtime.hitl import wrap_with_confirmation

    async def _run(target: str) -> str:
        return "ok"

    base = StructuredTool.from_function(coroutine=_run, name="dangerous_do", description="d")
    assert not getattr(base, "hitl_guarded", False)
    assert getattr(wrap_with_confirmation(base, "some_skill"), "hitl_guarded", False)


def test_tool_work_bounded_but_human_wait_unbounded():
    """Ключевое свойство: РАБОТА тула ограничена дедлайном (зависший не висит вечно —
    регресс-набор поймал ход на 8.5ч), а ОЖИДАНИЕ ЧЕЛОВЕКА из дедлайна вычитается."""
    from src.graph.agent import _await_tool_work_bounded

    async def _hung():                                  # «работает» дольше дедлайна
        await asyncio.sleep(5)
        return "поздно"

    async def _thinking_human():                        # всё время — раздумья человека
        with runbudget.human_pause():
            await asyncio.sleep(2.5)
        return "ответ юзера"

    async def _run():
        with run_context.request_scope("run_bound", "u"):
            runbudget.reset()
            with pytest.raises(asyncio.TimeoutError):    # зависший тул обрывается
                await _await_tool_work_bounded(_hung(), deadline=1.0, poll=0.2)
            runbudget.reset()
            # тот же дедлайн 1с, но 2.5с — ожидание человека → НЕ обрывается
            assert await _await_tool_work_bounded(
                _thinking_human(), deadline=1.0, poll=0.2) == "ответ юзера"

    asyncio.run(_run())


def test_clarify_asks_once_per_thread():
    """Второй раунд уточнений подряд НЕ задаётся: агент работает с допущениями (валидация:
    4 хода подряд «в каком формате?» и игнор «давай уже делай»)."""
    from src.graph.agent import _CLARIFY_MARK, _clarify_already_asked

    fresh = {"chat_history": [{"role": "user", "content": "помоги с отчётом"}]}
    assert not _clarify_already_asked(fresh)
    after = {"chat_history": [
        {"role": "user", "content": "помоги с отчётом"},
        {"role": "assistant", "content": f"{_CLARIFY_MARK}, пожалуйста:\n1. Тема?"},
        {"role": "user", "content": "по продажам за квартал"},
    ]}
    assert _clarify_already_asked(after)          # уже спрашивали → второй раз не переспрашиваем
    assert not _clarify_already_asked({"chat_history": []})


def test_clarify_marker_matches_formatter():
    """Маркер и формат вопросов не должны разъехаться (иначе guard перестанет ловить)."""
    from src.graph.agent import _CLARIFY_MARK, _format_clarify_questions

    assert _CLARIFY_MARK in _format_clarify_questions([{"question": "Что?", "options": []}])


def test_no_channel_still_short_circuits():
    """Канала нет — ждать некого: вопросы возвращаются как ответ (не виснем вечно впустую)."""
    from src.interface import clarify

    clarify.set_clarifier(None)
    assert not clarify.has_channel()
