"""recall-гейт («recall не всегда должен быть»): ГИБКАЯ память — и факты, и ассоциативная
память (эпизоды/выводы) отбираются ПО РЕЛЕВАНТНОСТИ к запросу (а не «персона всегда»).
Иначе task-факты одной задачи текут в несвязанный запрос. Без эмбеддингов (token-overlap)."""
import pytest

from src.memory.store import MemoryStore

UID = "gate-user"


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "memory.db"))
    yield s
    s.close()


def _seed(store):
    store.add_fact(UID, "язык_общения", "русский", importance=0.9, tags=["персона"])
    store.add_episode(UID, "найди трек purple hearts и включи его", "Включаю трек.",
                      outcome="ok", confidence=0.8, mode="deliberate")


def test_gate_drops_unrelated_facts_and_associative(store):
    _seed(store)
    # ГИБКАЯ память: нерелевантный запрос → отсекаются И ассоциативная память, И нерелевантные
    # факты (task-факт одной задачи не течёт в несвязанный запрос — анти-лик).
    text, score = store.recall_scored(UID, "квантовая хромодинамика глюоны конфайнмент", gate=0.22)
    assert "язык_общения" not in text               # факт нерелевантен запросу → не инжектится
    assert "Похожие прошлые задачи" not in text     # эпизоды отсечены гейтом
    assert score < 0.22


def test_gate_keeps_relevant_fact(store):
    # Факт, релевантный запросу, инжектится (гибкий retrieval, не «всегда» и не «никогда»).
    store.add_fact(UID, "любимый_фреймворк", "django python веб разработка", importance=0.5)
    text, _ = store.recall_scored(UID, "помоги с django python веб проектом", gate=0.22)
    assert "любимый_фреймворк" in text


def test_gate_keeps_relevant_associative(store):
    _seed(store)
    # Релевантный запрос → ассоциативная память присутствует.
    text, score = store.recall_scored(UID, "включи purple hearts", gate=0.22)
    assert "Похожие прошлые задачи" in text
    assert score >= 0.22


def test_gate_off_keeps_all(store):
    _seed(store)
    # gate=0 → старое поведение: эпизоды инжектятся независимо от релевантности (recency).
    text, _ = store.recall_scored(UID, "квантовая хромодинамика глюоны", gate=0.0)
    assert "Похожие прошлые задачи" in text


def test_recall_backward_compat_returns_str(store):
    _seed(store)
    out = store.recall(UID, "кто я такой")  # старая сигнатура → str (тул/тесты)
    assert isinstance(out, str)
    assert "язык_общения" in out


def test_query_embedded_once(monkeypatch, tmp_path):
    """Анти-регресс стоимости: recall эмбеддит запрос ОДИН раз, не на каждый кандидат."""
    calls = {"n": 0}

    class _CountingEmbedder:
        enabled = True

        def embed(self, text):
            calls["n"] += 1
            # детерминированный псевдо-вектор по длине токенов (косинус не важен для счёта)
            return [float(len(text) % 7 + 1), 1.0, 0.5]

    s = MemoryStore(str(tmp_path / "m.db"), embedder=_CountingEmbedder())
    for i in range(6):
        s.add_fact(UID, f"факт{i}", f"значение {i}", importance=0.5)
    for i in range(4):
        s.add_episode(UID, f"эпизод запрос {i}", "ответ", outcome="ok", confidence=0.7, mode="deliberate")
    calls["n"] = 0
    s.recall_scored(UID, "какой-то запрос", gate=0.0)
    # При 6 фактах + 4 эпизодах старый код эмбеддил бы запрос ~10+ раз; теперь — ровно 1.
    assert calls["n"] == 1, f"запрос эмбеддился {calls['n']} раз вместо 1"
    s.close()
