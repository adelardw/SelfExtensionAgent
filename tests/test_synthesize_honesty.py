"""synthesize не должен заявлять «выполнено», если прогон обрезан бюджетом или шаги не
завершены (живой баг: «я добавил в корзину» при 0% уверенности и исчерпанном бюджете)."""
import os
import pytest

needs_key = pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="llm строится на импорте src.agent",
)


@needs_key
def test_synthesize_refuses_false_completion(monkeypatch):
    import asyncio
    import src.agent as A

    class _Resp:
        content = "Я добавил в корзину пиццу с морепродуктами и одну любую пиццу."

    class _Chain:
        async def ainvoke(self, _): return _Resp()

    monkeypatch.setattr(A, "synth_chain", _Chain())
    monkeypatch.setattr(A.runbudget, "exhausted", lambda *a, **k: True)  # бюджет исчерпан
    state = {"query": "закажи пиццу", "subtasks": [
        {"goal": "добавить пиццу", "status": "partial"}], "step_results": [
        {"goal": "добавить пиццу", "result": "открыл меню"}]}
    out = asyncio.run(A.synthesize_node(state))
    ans = out["final_answer"].lower()
    assert "не довёл" in ans or "не довел" in ans  # честно, не «добавил»
    assert "проверь" in ans


@needs_key
def test_synthesize_keeps_success_when_complete(monkeypatch):
    import asyncio
    import src.agent as A

    class _Resp:
        content = "Готово, всё добавил в корзину."

    class _Chain:
        async def ainvoke(self, _): return _Resp()

    monkeypatch.setattr(A, "synth_chain", _Chain())
    monkeypatch.setattr(A.runbudget, "exhausted", lambda *a, **k: False)  # бюджет ок
    state = {"query": "закажи пиццу", "subtasks": [
        {"goal": "добавить пиццу", "status": "done"}], "step_results": [
        {"goal": "добавить пиццу", "result": "в корзине"}]}
    out = asyncio.run(A.synthesize_node(state))
    assert "Готово" in out["final_answer"]  # всё done → не трогаем
