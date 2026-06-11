"""Gap 2: измеримый before/after accept-гейт (без живых LLM-вызовов)."""
import pytest

from src.improve.pipe import SelfLearningPipe, _ABVerdict, _fill_placeholders


@pytest.fixture
def store(tmp_path):
    from src.memory.store import MemoryStore
    return MemoryStore(str(tmp_path / "m.db"))


def _fake_chat(verdict_better):
    class _R:
        content = "сгенерированный ответ"

    class _Judge:
        def invoke(self, _s):
            return _ABVerdict(better=verdict_better, reason="t")

    class _Model:
        def invoke(self, _msgs):
            return _R()

        def with_structured_output(self, _schema):
            return _Judge()

    return lambda *a, **k: _Model()


def test_fill_placeholders_removes_braces():
    assert "{" not in _fill_placeholders("текст {memory_context} и {chat_history} конец")


def test_before_after_accepts_on_improvement(store, monkeypatch):
    monkeypatch.setattr("src.improve.pipe.chat", _fake_chat("new"))
    pipe = SelfLearningPipe(store)
    fails = [{"query": f"кейс {i}", "feedback": "плохо"} for i in range(3)]
    ab = pipe._before_after_eval("старый промпт {x}", "новый промпт {x}", fails)
    assert ab["after"] == 3 and ab["before"] == 0 and ab["improved"] is True


def test_before_after_rejects_when_not_better(store, monkeypatch):
    monkeypatch.setattr("src.improve.pipe.chat", _fake_chat("old"))
    pipe = SelfLearningPipe(store)
    fails = [{"query": f"кейс {i}", "feedback": "плохо"} for i in range(3)]
    ab = pipe._before_after_eval("старый", "новый", fails)
    assert ab["improved"] is False  # новый НЕ лучше → откат (не сохраняем)
