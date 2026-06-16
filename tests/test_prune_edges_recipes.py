"""
prune() чистит повисшие рёбра графа и капает рецепты (долг ревью 2e).

Раньше `prune` обслуживал только episodes/facts/reflections; `memory_edges` и `recipes` росли без
границ → `neighbors()` замедлялся, БД пухла. Тест проверяет: повисшие рёбра удаляются, ЖИВЫЕ —
сохраняются, рецепты обрезаются до капа (по ценности).
"""
import tempfile
from pathlib import Path

from src.memory.store import MemoryStore


def _store() -> MemoryStore:
    return MemoryStore(str(Path(tempfile.mkdtemp()) / "m.db"))


def _count(s: MemoryStore, table: str) -> int:
    return s._conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]


def test_prune_removes_dangling_edges_keeps_live():
    s = _store()
    try:
        e = [s.add_episode("u", f"q{i}", "a") for i in range(3)]
        s.add_edge("u", "episode", e[0], "episode", e[1])   # живое
        s.add_edge("u", "episode", e[1], "episode", e[2])   # живое
        s.add_edge("u", "episode", e[0], "episode", 999999)  # повисшее (dst отсутствует)

        assert _count(s, "memory_edges") == 3
        removed = s.prune()
        assert removed["edges"] == 1
        assert _count(s, "memory_edges") == 2                # живые рёбра целы
    finally:
        s.close()


def test_prune_caps_recipes_by_value():
    s = _store()
    try:
        queries = ["погода алматы", "курс доллара", "рецепт борща", "код питон баг", "музыка джаз"]
        for q in queries:
            s.add_recipe("u", q, ["web_search"], [{"goal": q, "kind": "research"}])
        assert _count(s, "recipes") == 5

        removed = s.prune(max_recipes=3)
        assert removed["recipes"] == 2
        assert _count(s, "recipes") == 3
    finally:
        s.close()


def test_prune_edges_after_episode_eviction():
    """Рёбра на эпизоды, вытесненные капом max_episodes, тоже становятся повисшими и чистятся."""
    s = _store()
    try:
        ids = [s.add_episode("u", f"q{i}", "a") for i in range(5)]
        s.add_edge("u", "episode", ids[0], "episode", ids[1])  # оба попадут под вытеснение
        s.add_edge("u", "episode", ids[3], "episode", ids[4])  # оба свежие — переживут
        removed = s.prune(max_episodes=2)                       # оставить 2 свежих эпизода
        assert removed["episodes"] == 3
        assert removed["edges"] == 1                            # ребро на вытесненные эпизоды убрано
        assert _count(s, "memory_edges") == 1
    finally:
        s.close()
