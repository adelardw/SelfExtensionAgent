"""Универсальный embedding-роутер интентов (src/intent.py): мультиязычная классификация,
рост из фидбек-лупа, деградация в None (→ регэксп-fallback). Мок-эмбеддер, без сети."""
import pytest

from src import intent


class _FakeEmb:
    """Детерминированный эмбеддер: one-hot по ключевому слову — имитирует мультиязычную
    близость (RU и EN синонимы → один и тот же вектор)."""
    enabled = True

    def embed(self, text):
        t = (text or "").lower()
        if any(k in t for k in ("buy", "купить", "price", "цен", "стоит", "сколько стоит", "адрес", "address",
                                "новост", "news", "лучш", "best", "оформ", "apply", "near me")):
            return [1.0, 0.0, 0.0, 0.0]   # web_grounding
        if any(k in t for k in ("open", "открой", "cart", "корзин", "log in", "залог", "кабинет", "account", "click", "нажми", "зайди")):
            return [0.0, 1.0, 0.0, 0.0]   # physical_browser
        if any(k in t for k in ("play", "включи", "поставь", "music", "музык", "трек", "фильм", "movie", "видео", "song", "запусти")):
            return [0.0, 0.0, 1.0, 0.0]   # play_media
        return [0.0, 0.0, 0.0, 1.0]       # self_contained


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setattr(intent, "CODEBOOK_FILE", tmp_path / "codebook.json")
    r = intent.IntentRouter()
    r._embedder = _FakeEmb()
    return r


def test_classify_multilingual(router):
    assert router.classify("where can I buy a laptop")["label"] == "web_grounding"
    assert router.classify("хочу купить телефон подешевле")["label"] == "web_grounding"  # RU → тот же класс
    assert router.classify("open the website and log in")["label"] == "physical_browser"
    assert router.classify("включи песню radiohead")["label"] == "play_media"
    assert router.classify("explain how recursion works")["label"] == "self_contained"


def test_disabled_returns_none(router):
    router._embedder = type("E", (), {"enabled": False, "embed": lambda self, t: None})()
    assert router.classify("где купить ноутбук") is None   # нет эмбеддингов → fallback на регэксп


def test_empty_codebook_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(intent, "CODEBOOK_FILE", tmp_path / "cb.json")
    r = intent.IntentRouter()

    class _NoSeed(_FakeEmb):
        def embed(self, text):
            return None  # seed не наполнится → пустой кодбук
    r._embedder = _NoSeed()
    assert r.classify("что угодно") is None


def test_add_exemplar_grows_and_dedups(router):
    router.classify("warmup")  # триггерим seed-загрузку
    before = len(router._entries)
    router.add_exemplar("закажи доставку еды на дом", "physical_browser")
    router.add_exemplar("закажи доставку еды на дом", "physical_browser")  # дедуп
    learned = [e for e in router._entries if e.get("learned")]
    assert len(learned) == 1
    assert len(router._entries) == before + 1


def test_cap_per_label(router):
    router.classify("warmup")
    for i in range(intent.MAX_PER_LABEL + 10):
        router.add_exemplar(f"buy item number {i} cheap", "web_grounding")
    learned = [e for e in router._entries if e.get("learned") and e["label"] == "web_grounding"]
    assert len(learned) <= intent.MAX_PER_LABEL


def test_persistence(router, tmp_path, monkeypatch):
    router.classify("warmup")
    router.add_exemplar("открой мой банк-аккаунт", "physical_browser")
    # новый роутер с тем же файлом — экземпляр должен подняться
    r2 = intent.IntentRouter()
    r2._embedder = _FakeEmb()
    r2._load()
    assert any(e.get("learned") and "банк" in e.get("text", "") for e in r2._entries)
