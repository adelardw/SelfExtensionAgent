"""Согласованность реестра обучаемых промптов и карты OPTIMIZABLE."""
from src.improve.graph_learn import OPTIMIZABLE
from src.llm.prompts import OPTIMIZABLE_PROMPTS, step_execution_system_prompt


def test_optimizable_roles_have_defaults():
    """Каждая роль из карты graph_learn должна иметь дефолтный промпт в реестре."""
    for node, role in OPTIMIZABLE.items():
        assert role in OPTIMIZABLE_PROMPTS, f"нет дефолта для роли '{role}' (нода {node})"
        assert OPTIMIZABLE_PROMPTS[role].strip(), f"пустой дефолт для '{role}'"


def test_researcher_is_optimizable():
    assert "researcher" in OPTIMIZABLE_PROMPTS


def test_step_execution_placeholders():
    """Критичные плейсхолдеры step_execution не должны пропасть при правках промпта."""
    for ph in ("{fewshots}", "{capability_hint}"):
        assert ph in step_execution_system_prompt, f"потерян плейсхолдер {ph}"
