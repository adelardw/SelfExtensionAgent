"""Универсальный embedding-роутер интентов (src/intent.py): мультиязычная классификация,
рост из фидбек-лупа, деградация в None (→ регэксп-fallback). Мок-эмбеддер, без сети."""
import pytest

from src.graph import intent


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


# Контролируемый seed: каждая фраза чисто маппится мок-эмбеддером в свой класс (тест логики
# роутера развязан от СОДЕРЖИМОГО реального _SEED — обогащение seed не ломает тесты).
_TEST_SEED = {
    "web_grounding": ["where to buy cheap stuff", "где купить дёшево"],
    "physical_browser": ["open site and log in", "открой сайт и залогинься"],
    "play_media": ["play music", "включи музыку"],
    "self_contained": ["explain concept", "объясни понятие"],
}


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setattr(intent, "CODEBOOK_FILE", tmp_path / "codebook.json")
    monkeypatch.setattr(intent, "_SEED", _TEST_SEED)
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


def test_model_change_invalidates_codebook(tmp_path, monkeypatch):
    import json as _j
    monkeypatch.setattr(intent, "CODEBOOK_FILE", tmp_path / "cb.json")

    class _E1(_FakeEmb):
        model = "model-A"

    class _E2(_FakeEmb):
        model = "model-B"

    r1 = intent.IntentRouter(); r1._embedder = _E1(); r1.classify("warmup")
    saved = _j.loads((tmp_path / "cb.json").read_text())
    assert saved["model"] == "model-A" and saved["entries"]
    # другой эмбеддер → кодбук инвалидируется и пере-сидится в НОВОМ пространстве
    r2 = intent.IntentRouter(); r2._embedder = _E2(); r2._load()
    saved2 = _j.loads((tmp_path / "cb.json").read_text())
    assert saved2["model"] == "model-B" and saved2["entries"]


def test_route_corpus_logs_pos_and_neg(tmp_path, monkeypatch):
    # Корпус для будущего contrastive: позитивы И негативы (reward 0/1), append-only.
    monkeypatch.setattr(intent, "_CORPUS_DB", str(tmp_path / "corpus.db"))
    monkeypatch.setattr(intent, "_corpus_conn", None)
    intent.log_route_example("где купить ноут", "web_grounding", 1, "u1")
    intent.log_route_example("включи музыку", "play_media", 0, "u1")     # негатив
    intent.log_route_example("", "web_grounding", 1)                      # пустой текст → игнор
    intent.log_route_example("x", "bad_label", 1)                         # неизвестный лейбл → игнор
    st = intent.corpus_stats()
    assert st["total"] == 2 and st["pos"] == 1 and st["neg"] == 1
    assert st["by_route"]["play_media"]["neg"] == 1


def test_persistence(router, tmp_path, monkeypatch):
    router.classify("warmup")
    router.add_exemplar("открой мой банк-аккаунт", "physical_browser")
    # новый роутер с тем же файлом — экземпляр должен подняться
    r2 = intent.IntentRouter()
    r2._embedder = _FakeEmb()
    r2._load()
    assert any(e.get("learned") and "банк" in e.get("text", "") for e in r2._entries)
