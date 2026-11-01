import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Тесты запускаются из корня репо (config.yml и src/ резолвятся относительно).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


# ── Детерминированный intent-роутер для тестов ──────────────────────────────────
# Роутинг агента теперь embedding-классификатор (мультиязычный, без регэкспов). В unit-тестах
# мокаем его keyword-фейком: тесты проверяют ПРОВОДКУ (label роутера → решение функции), а
# реальную ТОЧНОСТЬ классификатора меряет src/eval/route_eval.py (живой, статистический). Это
# делает suite герметичным (без сетевых эмбеддинг-вызовов). test_intent.py использует свои
# инстансы IntentRouter напрямую (get_router не зовёт) — на него фейк не влияет.
class _DetRouter:
    enabled = True
    _KW = [
        ("play_media", ("включи", "трек", "музык", "песн", "play ", "фильм", "видео",
                        "плейлист", "сыграй", "поставь фильм")),
        ("physical_browser", ("открой", "браузер", "скриншот", "корзин", "войди", "залог",
                              "кликни", "нажми", "заполни", "паузу", "останови", "оформить заказ")),
        ("web_grounding", ("где", "лучш", "адрес", "суши", "купить", "пособи", "кофейн",
                          "стоит", "загран", "выбрать", "подарить", "обзор", "наушник",
                          "iphone", "samsung", "пылесос", "ноутбук", "брюк", "цен", "price",
                          "buy", "best", "купи")),
    ]

    def classify(self, text, qvec=None):
        t = (text or "").lower()
        for label, kws in self._KW:
            if any(k in t for k in kws):
                return {"label": label, "score": 0.9, "scores": {}}
        return {"label": "self_contained", "score": 0.9, "scores": {}}


@pytest.fixture(autouse=True)
def _det_intent_router(monkeypatch):
    import src.graph.intent as _intent
    monkeypatch.setattr(_intent, "get_router", lambda: _DetRouter())


def pytest_configure(config):
    config.addinivalue_line("markers", "live: opt-in semantic/API evaluation (SEA_LIVE_TESTS=1)")


def pytest_collection_modifyitems(config, items):
    import os
    if os.getenv("SEA_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live evaluation: set SEA_LIVE_TESTS=1 with working credentials")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def offline_services(monkeypatch):
    """Unit tests exercise wiring, not a paid classifier. Real semantics are opt-in."""
    import os
    if os.getenv("SEA_LIVE_TESTS") == "1":
        return
    import socket
    from src.improve import safety
    from src.graph.semantic_signals import _ContrastiveSignal

    class StubEmb:
        enabled = True

        def embed(self, text):
            text = text.lower()
            meta = any(x in text for x in ("article", "explains", "educational", "owasp", "статья", "защищаться"))
            attack = not meta and any(x in text for x in (
                "ignore", "reveal", "обойди", "игнорируй", "jailbreak", "забудь", "disable", "сними", "忽略", "bypass"))
            return [1., 0.] if attack else [0., 1.]

    monkeypatch.setattr(safety, "_INJ_DETECTOR", _ContrastiveSignal(safety._INJ_POS, safety._INJ_NEG, StubEmb()))
    monkeypatch.setattr(safety, "_verdict_cache", {})
    monkeypatch.setattr(safety, "_offline_noted", False)

    def no_network(*args, **kwargs):
        raise RuntimeError("network disabled in unit tests; mock transport or mark test live")

    monkeypatch.setattr(socket.socket, "connect", no_network)
