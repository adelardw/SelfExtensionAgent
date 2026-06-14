"""Анти-ложь-о-завершении перенесена из synthesize-регэкспа («добавил|заказал…») в финальный
LLM-валидатор: судья получает run_status=incomplete и флагает ValidationResult.false_completion
семантически (любой язык), а validation_node подменяет честным статусом с прогрессом (живой баг:
«я добавил в корзину» при 0% уверенности и исчерпанном бюджете). Тут тестируется ЛОГИКА АГЕНТА
на флаг — судья замокан (без сети). Импорт src.agent строит LLM → @needs_key."""
import os
import pytest

needs_key = pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="llm строится на импорте src.agent",
)


def _judge(**kw):
    """Замоканный финальный судья: возвращает ValidationResult с заданными флагами."""
    from src.structured_outputs import ValidationResult

    class _Chain:
        async def ainvoke(self, _payload):
            return ValidationResult(is_valid=True, confidence=0.8, feedback="ок", **kw)

    return _Chain()


@needs_key
def test_validator_flags_false_completion(monkeypatch):
    """Оборванный прогон + судья поднял false_completion → честный статус с прогрессом, ПРИНЯТЬ."""
    import asyncio
    import src.agent as A

    monkeypatch.setattr(A, "validation_chain", _judge(false_completion=True))
    monkeypatch.setattr(A, "CONSENSUS_VALIDATION", False)
    monkeypatch.setattr(A.runbudget, "exhausted", lambda *a, **k: True)  # прогон оборван бюджетом
    state = {"query": "закажи пиццу",
             "final_answer": "Я добавил в корзину пиццу с морепродуктами и одну любую пиццу.",
             "subtasks": [{"goal": "добавить пиццу", "status": "partial"}]}
    out = asyncio.run(A.validation_node(state))
    ans = out["final_answer"].lower()
    assert "не довёл" in ans or "не довел" in ans   # честно, не «добавил»
    assert "проверь" in ans
    assert out["validation_passed"] is True


@needs_key
def test_validator_keeps_answer_when_complete(monkeypatch):
    """Прогон завершён (все шаги done, бюджет цел) → флаг не применяется: ответ НЕ подменяем."""
    import asyncio
    import src.agent as A

    monkeypatch.setattr(A, "validation_chain", _judge(false_completion=False))
    monkeypatch.setattr(A, "CONSENSUS_VALIDATION", False)
    monkeypatch.setattr(A.runbudget, "exhausted", lambda *a, **k: False)  # бюджет ок
    state = {"query": "закажи пиццу",
             "final_answer": "Готово, всё добавил в корзину.",
             "subtasks": [{"goal": "добавить пиццу", "status": "done"}]}
    out = asyncio.run(A.validation_node(state))
    assert "final_answer" not in out          # ответ не тронут (нет подмены)
    assert out["validation_passed"] is True


@needs_key
def test_false_completion_guarded_by_incomplete(monkeypatch):
    """Даже если судья ошибочно поднял false_completion на ЗАВЕРШЁННОМ прогоне — структурный
    гейт incomplete не даёт подменить ответ (поле только про оборванный прогон)."""
    import asyncio
    import src.agent as A

    monkeypatch.setattr(A, "validation_chain", _judge(false_completion=True))
    monkeypatch.setattr(A, "CONSENSUS_VALIDATION", False)
    monkeypatch.setattr(A.runbudget, "exhausted", lambda *a, **k: False)  # прогон НЕ оборван
    state = {"query": "закажи пиццу",
             "final_answer": "Готово, добавил в корзину.",
             "subtasks": [{"goal": "добавить пиццу", "status": "done"}]}  # всё done → complete
    out = asyncio.run(A.validation_node(state))
    assert "final_answer" not in out          # incomplete=False → подмены нет
