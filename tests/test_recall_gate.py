"""recall-гейт («recall не всегда должен быть»): персона-факты — всегда, ассоциативная
память (эпизоды/выводы) — только при релевантности запросу. Без эмбеддингов (token-overlap)."""
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


def test_gate_drops_unrelated_associative_keeps_persona(store):
    _seed(store)
    # Нерелевантный запрос → ассоциативная память отсекается, персона остаётся.
    text, score = store.recall_scored(UID, "квантовая хромодинамика глюоны конфайнмент", gate=0.22)
    assert "язык_общения" in text                  # персона ВСЕГДА
    assert "Похожие прошлые задачи" not in text     # эпизоды отсечены гейтом
    assert score < 0.22


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
