"""Heavy-режим: маршрутизация «сборка → сквозной ревью → доработка» + temp-навыки."""
import os

import pytest

needs_key = pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="нужен API-ключ: llm строится на импорте src.agent",
)


@needs_key
def test_heavy_routing():
    from src.graph.agent import (
        MAX_REVISIONS,
        route_after_goal,
        route_after_reflexion,
        route_after_review,
        route_after_synthesize,
    )

    # heavy (если задан force_mode) идёт по deliberate-пути (goal → router), не в fast_answer
    assert route_after_reflexion({"mode": "heavy"}) == "goal"
    assert route_after_goal({"mode": "heavy"}) == "router"

    # Thread 3c: сквозной ревью — ЗАРАБОТАННАЯ эскалация (артефакт большой+многошаговый+rubric),
    # а НЕ предсказанный режим heavy. Дешёвые рантайм-сигналы, не догадка модели.
    earned = {"final_answer": "x" * 1500, "goal_rubric": ["c1"],
              "subtasks": [{"status": "done"} for _ in range(3)], "revision_rounds": 0}
    assert route_after_synthesize(earned) == "review"
    # маленький артефакт / мало шагов → ревью НЕ заработан → сразу validation
    assert route_after_synthesize({"final_answer": "коротко", "revision_rounds": 0}) == "validation"
    # заработан, но бюджет раундов исчерпан → validation
    assert route_after_synthesize({**earned, "revision_rounds": MAX_REVISIONS}) == "validation"
    # явный force_mode='heavy' (юзер потребовал тщательность) → review даже без большого артефакта
    assert route_after_synthesize({"force_mode": "heavy", "revision_rounds": 0}) == "review"
    assert route_after_synthesize({"mode": "deliberate", "revision_rounds": 0}) == "validation"

    # ревью добавил fix-подшаги → обратно в шаговый цикл; чисто → validation
    two_steps = [{"goal": "a"}, {"goal": "b"}]
    assert route_after_review({"current_step": 1, "subtasks": two_steps}) == "step_executor"
    assert route_after_review({"current_step": 2, "subtasks": two_steps}) == "validation"


@needs_key
def test_review_node_in_graph():
    from src.graph.agent import build_graph

    nodes = set(build_graph().get_graph().nodes)
    assert "review" in nodes


def test_temp_skill_lifecycle(tmp_path, monkeypatch):
    from src.tools import skill_creation as sc

    monkeypatch.setattr(sc, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(sc, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(sc, "PROTECTED_SKILLS", set())

    (tmp_path / "one_off").mkdir()
    sc._save_registry({"one_off": {"description": "x", "created_at": "2020-01-01T00:00:00"}})

    sc.mark_temporary("one_off")
    assert sc._load_registry()["one_off"]["temporary"] is True
    assert "🕒temp" in sc.list_skills.invoke({})

    # принятие в библиотеку снимает флаг
    sc.clear_temporary("one_off")
    assert "temporary" not in sc._load_registry()["one_off"]

    # протухший temp-навык вычищается sync_registry (created_at 2020 — старше TTL)
    sc.mark_temporary("one_off")
    report = sc.sync_registry()
    assert "one_off" in report["expired_temp"]
    assert "one_off" not in sc._load_registry()
    assert not (tmp_path / "one_off").exists()


@needs_key
def test_llm_tiers():
    """Роли fast/code/deep — РАЗНЫЕ модели по config.yml (тиры). Не хардкодим имена,
    чтобы тест не падал при смене моделей в конфиге."""
    from omegaconf import OmegaConf

    from src.llm.llm import model_for

    cfg = OmegaConf.load("config.yml")
    assert model_for("fast") == cfg.model.name
    assert model_for("code") == cfg.code_model.name
    assert model_for("deep") == cfg.deep_model.name
    # тиры действительно различаются (роутинг ≠ исполнение ≠ deep-ревью)
    assert len({model_for("fast"), model_for("code"), model_for("deep")}) >= 2
