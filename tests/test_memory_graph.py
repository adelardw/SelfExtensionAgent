"""GraphRAG-lite: densify рёбер fact↔fact по cosine (без LLM) + spreading-activation в recall
(ассоциативный пул: факт, релевантный ЧЕРЕЗ связь с релевантным эпизодом, а не лексически)."""
import pytest

from src.memory.store import MemoryStore

UID = "graph-user"


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "m.db"))
    yield s
    s.close()


class _KeywordEmbedder:
    """Детерминированный эмбеддер: вектор по ключевому слову (для проверки densify-cosine)."""
    enabled = True

    def embed(self, text):
        t = (text or "").lower()
        if "python" in t:
            return [1.0, 0.0, 0.0]
        if "java" in t:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_densify_links_similar_facts(tmp_path):
    s = MemoryStore(str(tmp_path / "d.db"), embedder=_KeywordEmbedder())
    f1 = s.add_fact(UID, "стек_бэкенд", "python и django")
    f2 = s.add_fact(UID, "любимый_язык", "python всегда")   # семантически близок f1
    f3 = s.add_fact(UID, "другое", "java и spring")          # далёк
    nb_ids = {(r["type"], r["id"]) for r in s.neighbors(UID, "fact", f1)}
    assert ("fact", f2) in nb_ids       # python↔python — densify-ребро есть
    assert ("fact", f3) not in nb_ids   # python↔java — ребра нет
    s.close()


def test_no_densify_without_embeddings(store):
    # NullEmbedder → векторов нет → densify не создаёт рёбер (graceful).
    f1 = store.add_fact(UID, "a", "python django")
    store.add_fact(UID, "b", "python flask")
    assert store.neighbors(UID, "fact", f1) == []


def test_graph_boost_lifts_connected_fact(store):
    """Spreading-activation: факт, связанный с РЕЛЕВАНТНЫМ эпизодом, ранжируется выше
    несвязанного — хотя оба лексически нерелевантны запросу."""
    ep = store.add_episode(UID, "проект rust бэкенд", "ок", outcome="ok",
                           confidence=0.8, mode="deliberate")
    fA = store.add_fact(UID, "альфа_контакт", "значение икс", importance=0.5)   # будет связан
    fB = store.add_fact(UID, "бета_контакт", "значение игрек", importance=0.5)  # НЕ связан
    store.add_edge(UID, "episode", ep, "fact", fA, relation="derived")

    q = "проект rust"  # релевантен эпизоду (сид), нерелевантен обоим фактам
    top_eps = store._rank_episodes(UID, q, 5)
    boost = store._graph_boost(UID, q, None, top_eps)
    assert ("fact", fA) in boost and boost[("fact", fA)] > 0
    ranked = store._rank_facts(UID, q, None, boost)
    ids = [f["id"] for f in ranked]
    assert ids.index(fA) < ids.index(fB)   # связанный выше несвязанного


def test_graph_no_pull_when_query_irrelevant(store):
    """PII-контейнмент: нерелевантный запрос не делает эпизод сидом → связанное НЕ тянется."""
    ep = store.add_episode(UID, "проект rust бэкенд", "ок", outcome="ok",
                           confidence=0.8, mode="deliberate")
    fA = store.add_fact(UID, "секрет_контакт", "значение икс")
    store.add_edge(UID, "episode", ep, "fact", fA, relation="derived")
    # запрос ни о чём похожем → эпизод НЕ сид (rel < seed_min) → boost пуст
    boost = store._graph_boost(UID, "квантовая хромодинамика глюоны",
                               None, store._rank_episodes(UID, "квантовая хромодинамика глюоны", 5))
    assert ("fact", fA) not in boost


def test_graph_hops_zero_disables(tmp_path):
    s = MemoryStore(str(tmp_path / "z.db"), graph_hops=0)
    ep = s.add_episode(UID, "проект rust", "ок", outcome="ok", confidence=0.8, mode="deliberate")
    fA = s.add_fact(UID, "k", "v")
    s.add_edge(UID, "episode", ep, "fact", fA, relation="derived")
    assert s._graph_boost(UID, "проект rust", None, s._rank_episodes(UID, "проект rust", 5)) == {}
    s.close()
