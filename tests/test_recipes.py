"""Рецепты (ступень-0 амортизации): успешный план компилируется, похожая задача его
переиспользует, проигрывающий рецепт самоудаляется. Офлайн, без LLM."""
import json

import pytest

from src.memory.store import MemoryStore

UID = "test-user"
Q = "собери еженедельный отчёт по продажам из excel в pdf"
PLAN = [
    {"goal": "Прочитать данные из excel", "done_check": "Данные загружены", "kind": "direct"},
    {"goal": "Сформировать pdf-отчёт", "done_check": "Файл создан", "kind": "compose",
     "status": "done", "result": "мусор не должен попасть в рецепт"},
]


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "memory.db"))
    yield s
    s.close()


def test_add_find_recipe(store):
    rid = store.add_recipe(UID, Q, ["stash", "generate_pptx"], PLAN)
    assert rid > 0
    r = store.find_recipe(UID, "сделай отчёт по продажам за неделю из excel в pdf")
    assert r and r["id"] == rid
    plan = json.loads(r["plan"])
    assert [p["goal"] for p in plan] == ["Прочитать данные из excel", "Сформировать pdf-отчёт"]
    assert "result" not in plan[1] and "status" not in plan[1]  # план без runtime-мусора
    assert store.find_recipe(UID, "какая погода в алматы") is None  # чужой кластер
    assert store.find_recipe("other-user", Q) is None               # изоляция юзеров


def test_recipe_dedup_updates(store):
    rid1 = store.add_recipe(UID, Q, ["stash"], PLAN)
    rid2 = store.add_recipe(UID, "собери отчёт по продажам из excel в pdf",
                            ["stash", "web_search"], PLAN[:1])
    assert rid2 == rid1  # похожий запрос → обновление, не дубль
    r = store.get_recipe(rid1)
    assert json.loads(r["skills"]) == ["stash", "web_search"]
    assert len(json.loads(r["plan"])) == 1


def test_losing_recipe_self_deletes(store):
    rid = store.add_recipe(UID, Q, ["stash"], PLAN)
    store.recipe_feedback(rid, win=True)
    store.recipe_feedback(rid, win=False)
    assert store.get_recipe(rid) is not None  # 1/2 — ещё живёт
    store.recipe_feedback(rid, win=False)     # 1/3 < 0.5 → самоудаление
    assert store.get_recipe(rid) is None


def test_empty_plan_not_saved(store):
    assert store.add_recipe(UID, Q, ["stash"], []) == 0
    assert store.add_recipe(UID, Q, ["stash"], [{"done_check": "без goal"}]) == 0


needs_key = pytest.mark.skipif(
    not __import__("os").getenv("OPEN_ROUTER_API_KEY") and not __import__("os").getenv("OPENAI_API_KEY"),
    reason="нужен API-ключ: llm строится на импорте src.agent",
)


@needs_key
def test_decompose_zero_llm_on_high_sim_recipe(monkeypatch, tmp_path):
    """sim ≥ 0.7 → план из рецепта БЕЗ LLM-вызова decompose (артефакт ЗАМЕНЯЕТ работу)."""
    import asyncio
    import src.agent as A

    s = MemoryStore(str(tmp_path / "memory.db"))
    rid = s.add_recipe("u1", Q, ["file_operations"], PLAN)
    monkeypatch.setattr(A, "memory_store", s)

    async def _no_llm(*a, **kw):
        raise AssertionError("decompose не должен звать LLM при высокой похожести рецепта")
    monkeypatch.setattr(A, "_structured", _no_llm)
    out = asyncio.run(A.decompose_node({
        "query": Q, "user_id": "u1", "recipe_id": rid, "selected_skills": []}))
    assert [st["goal"] for st in out["subtasks"]] == ["Прочитать данные из excel", "Сформировать pdf-отчёт"]
    assert all(st["status"] == "pending" for st in out["subtasks"])
    assert "проверенный рецепт" in out["plan"]
    s.close()
