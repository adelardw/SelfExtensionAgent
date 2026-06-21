"""
Изоляция бюджета прогона по run_id (долг ревью 2a).

Раньше `_used`/`_start` были module-global: на сервере reset() одного запроса обнулял счётчик и
таймер ВСЕМ запросам в полёте. Теперь состояние живёт в `_states[run_id]`, run_id — в contextvar,
выставляемом на ГРАНИЦЕ запроса (`run_scope`). Ключевое свойство: scope, заданный во ВНЕШНЕМ
контексте, наследуется дочерними asyncio-задачами (ноды/callback) ВНИЗ, но конкурентные прогоны
изолированы. При этом без `run_scope` поведение прежнее (общий `_default`) — eval/REPL не затронуты.
"""
import asyncio

from src.runtime import runbudget as rb


def test_sequential_default_unchanged():
    """Без run_scope — общий _default, как до правки (eval/REPL последовательны)."""
    rb.reset()
    rb.add(100)
    rb.add(50)
    assert rb.used() == 150
    assert rb.exhausted(120, 1e9) is True
    assert rb.exhausted(200, 1e9) is False


def test_concurrent_runs_isolated():
    """Конкурентные прогоны не стирают счётчик друг другу (суть бага 2a)."""

    async def run(rid: str, tokens: list[int]) -> int:
        with rb.run_scope(rid):
            rb.reset()

            async def node_work() -> int:        # дочерняя задача = нода/callback
                for t in tokens:
                    rb.add(t)                    # add() резолвит run_id из contextvar (как callback)
                return rb.used()

            return await asyncio.create_task(node_work())

    async def main():
        return await asyncio.gather(
            run("A", [1000, 1000, 1000]),
            run("B", [10, 10]),
            run("C", [500]),
        )

    a, b, c = asyncio.run(main())
    assert (a, b, c) == (3000, 20, 500)          # каждый видит ТОЛЬКО свои токены


def test_run_scope_cleans_registry():
    with rb.run_scope("temp-run"):
        rb.reset()
        rb.add(42)
        assert rb.used() == 42
    assert "temp-run" not in rb._states           # состояние убрано по выходе


def test_armed_callback_aborts_when_over_budget():
    """Оживший предохранитель: вооружённый шаг + исчерпанный бюджет → BudgetExceeded из callback
    (раньше arm/disarm не вызывались нигде → мид-степ обрыва не было, докстринг врал)."""
    cb = rb.callback()
    assert cb.raise_error is True                    # иначе langchain проглотит исключение

    rb.reset()
    rb.add(10_000)
    cb.on_chat_model_start()                          # disarmed → тихо, даже при перерасходе

    rb.arm(token_limit=5_000, sec_limit=1e9)
    try:
        with __import__("pytest").raises(rb.BudgetExceeded):
            cb.on_chat_model_start()                  # armed + over → обрыв
    finally:
        rb.disarm()


def test_abort_propagates_through_langchain_ainvoke():
    """Не синтетика: обрыв реально пробивается через langchain .ainvoke (raise_error=True), а не
    проглатывается. Без сети — fake chat-модель с привязанным budget-callback."""
    import asyncio
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 10))
    model = model.with_config(callbacks=[rb.callback()])

    async def run() -> str:
        rb.reset()
        rb.add(10_000)
        rb.arm(token_limit=5_000, sec_limit=1e9)        # уже за бюджетом
        try:
            await model.ainvoke([HumanMessage(content="hi")])
            return "not_aborted"
        except rb.BudgetExceeded:
            return "aborted"
        finally:
            rb.disarm()

    assert asyncio.run(run()) == "aborted"


def test_on_llm_end_defensive_against_malformed_usage():
    """raise_error=True делает фатальной ЛЮБУЮ ошибку callback'а. Кривой usage (не dict) от
    нестандартного провайдера НЕ должен ронять прогон — это лишь телеметрия (острый край #A1)."""
    cb = rb.callback()

    class _FakeResp:
        llm_output = {"usage": "not-a-dict"}   # провайдер вернул строку вместо dict
        generations = []

    cb.on_llm_end(_FakeResp())                 # не должно бросить
    cb.on_llm_end(object())                    # совсем мусорный ответ — тоже тихо


def test_budget_exceeded_pierces_broad_except():
    """BudgetExceeded = BaseException → широкие `except Exception` (как в research.py) его НЕ глотают,
    он доходит до явного хендлера step_executor (острый край #A3, проверено вживую а не чтением)."""
    import pytest

    def research_like():
        try:
            raise rb.BudgetExceeded("explosion")
        except Exception:                      # как research.py — ловит обычные ошибки
            return "swallowed"

    with pytest.raises(rb.BudgetExceeded):
        research_like()                        # пробил насквозь, не «swallowed»


def test_armed_under_budget_does_not_abort():
    rb.reset()
    rb.add(100)
    rb.arm(token_limit=5_000, sec_limit=1e9)
    try:
        cb = rb.callback()
        cb.on_chat_model_start()                      # в бюджете → не кидает
    finally:
        rb.disarm()


def test_reset_does_not_stomp_other_run():
    """reset() одного прогона НЕ обнуляет счётчик другого (прямой репро 2a)."""

    async def main():
        with rb.run_scope("X"):
            rb.reset()
            rb.add(900)

            async def other_request():
                with rb.run_scope("Y"):
                    rb.reset()                    # раньше это обнуляло бы счётчик X
                    rb.add(5)
                    return rb.used()

            y_used = await asyncio.create_task(other_request())
            return rb.used(), y_used

    x_used, y_used = asyncio.run(main())
    assert x_used == 900 and y_used == 5
