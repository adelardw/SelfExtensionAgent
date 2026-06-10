"""Сборка графа и под-агентов после миграции на langchain.agents.create_agent."""
import os

import pytest

needs_key = pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="нужен API-ключ: llm строится на импорте src.agent",
)


@needs_key
def test_agent_module_imports_and_graph_builds():
    from src.agent import build_graph  # noqa: PLC0415

    graph = build_graph()
    assert graph is not None
    nodes = set(graph.get_graph().nodes)
    for expected in ("recall", "goal", "reflexion", "step_executor", "reflect"):
        assert expected in nodes, f"в графе нет ноды {expected}"


@needs_key
def test_researcher_subagent_builds():
    from src.subagents import get_subagent_tools  # noqa: PLC0415

    tools = get_subagent_tools(["researcher"])
    assert len(tools) == 1
    assert tools[0].name == "run_researcher"
