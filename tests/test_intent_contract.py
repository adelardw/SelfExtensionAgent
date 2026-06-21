"""
Контракт intent-роутинга при отсутствии сигнала (долг ревью: docstring обещал регэксп-fallback,
которого у потребителей нет).

Зафиксировано КОДОМ (а не прозой): classify() без эмбеддера → None; живые потребители из agent.py
при None возвращают False и полагаются на reflexion — НАМЕРЕННО без регэксп-костыля
(feedback «no-keyword-crutches»). Это характеризационный тест: ловит регресс, если кто-то добавит
регэксп-ветку или поменяет деградацию.
"""
from src.graph import intent


class _StubRouter:
    """Роутер без эмбеддера: всегда None (как реальный classify при embeddings off)."""

    def classify(self, text, qvec=None):
        return None


def test_classify_none_without_embeddings():
    r = intent.IntentRouter()
    # без ключа/эмбеддера enabled=False → classify обязан вернуть None, не падать
    if not r.enabled:
        assert r.classify("где купить кофе в Алматы") is None


def test_callers_defer_to_reflexion_on_none(monkeypatch):
    """Нет сигнала → helpers возвращают False (отдают решение reflexion), НЕ регэксп-эвристику."""
    from src.graph import agent

    monkeypatch.setattr(intent, "get_router", lambda: _StubRouter())

    # явно «веб-граундинг»-запрос: при None всё равно False — значит ветки на регэкспах нет
    assert agent._needs_web_grounding("где купить и сколько стоит iphone сейчас") is False
    assert agent._wants_physical_browser("включи музыку и поставь на паузу") is False
    assert agent._is_play_intent("включи трек на ютубе") is False


def test_positive_label_routes_true(monkeypatch):
    """Когда классификатор уверенно вернул label — helper срабатывает (контроль, что False не зашит)."""
    from src.graph import agent

    class _PosRouter:
        def classify(self, text, qvec=None):
            return {"label": "web_grounding", "score": 0.9, "scores": {}}

    monkeypatch.setattr(intent, "get_router", lambda: _PosRouter())
    assert agent._needs_web_grounding("любой запрос") is True
