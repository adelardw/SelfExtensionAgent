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
    import src.intent as _intent
    monkeypatch.setattr(_intent, "get_router", lambda: _DetRouter())
