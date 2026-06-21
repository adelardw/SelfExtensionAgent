"""
Под-агенты и под-графы как ИНСТРУМЕНТЫ (agent/graph-as-tool).

Базовый набор инструментов исполнителя единообразен: навык (python @tool),
MCP-инструмент, под-агент (ReAct внутри) и под-граф (скомпилированный LangGraph).
Всё это — обычные LangChain-tools в одном списке `tools=[...]`, поэтому
`decompose`/`step_executor` могут осознанно делегировать подшаг любому из них.

Аддитивно: основной граф не трогаем. Под-агенты вызываются как инструменты,
трейсятся (tools_attached) и их промпты — обучаемые (OPTIMIZABLE_PROMPTS).
"""
from __future__ import annotations

from typing import Callable, Optional

from langchain_core.tools import StructuredTool
from langchain.agents import create_agent

from src.improve.prompt_store import get_prompt
from src.llm.llm import chat
from src.llm.prompts import OPTIMIZABLE_PROMPTS
from src.tools import get_all_loaded_skill_tools


def subgraph_to_tool(name: str, description: str, compiled, in_key: str = "query",
                     out_key: str = "final_answer", recursion_limit: int = 25) -> StructuredTool:
    """Оборачивает скомпилированный LangGraph (под-граф) в инструмент."""
    async def _run(task: str) -> str:
        res = await compiled.ainvoke({in_key: task}, config={"recursion_limit": recursion_limit})
        return str(res.get(out_key, res)) if isinstance(res, dict) else str(res)

    return StructuredTool.from_function(coroutine=_run, name=name, description=description)


def react_subagent_tool(name: str, description: str, system: str,
                        skill_names: Optional[list[str]] = None, model: Optional[str] = None) -> StructuredTool:
    """Оборачивает ReAct-под-агента (со своим промптом и набором навыков) в инструмент."""
    llm = chat("code")
    tools = get_all_loaded_skill_tools(skill_names) if skill_names else []
    agent = create_agent(llm, tools, system_prompt=system)

    async def _run(task: str) -> str:
        r = await agent.ainvoke({"messages": [("user", task)]}, config={"recursion_limit": 25})
        msgs = r.get("messages", [])
        return msgs[-1].content if msgs else ""

    return StructuredTool.from_function(coroutine=_run, name=name, description=description)


# Каталог под-агентов (фабрики, чтобы строить лениво — без LLM на импорте).
# Промпт researcher — обучаемый параметр: дефолт в OPTIMIZABLE_PROMPTS,
# override берётся из ParamStore в момент сборки.
SUBAGENT_CATALOG: dict[str, Callable[[], StructuredTool]] = {
    "researcher": lambda: react_subagent_tool(
        "run_researcher",
        "Глубокий веб-исследователь (под-агент): сам ищет, читает страницы и синтезирует ответ с источниками.",
        get_prompt("researcher", OPTIMIZABLE_PROMPTS["researcher"]),
        skill_names=["web_search"],
    ),
}

_cache: dict[str, StructuredTool] = {}


def get_subagent_tools(names: Optional[list[str]] = None) -> list[StructuredTool]:
    """Возвращает под-агентов как инструменты (по умолчанию весь каталог)."""
    wanted = names or list(SUBAGENT_CATALOG)
    out = []
    for n in wanted:
        if n not in SUBAGENT_CATALOG:
            continue
        if n not in _cache:
            try:
                _cache[n] = SUBAGENT_CATALOG[n]()
            except Exception as e:  # noqa: BLE001
                print(f"[subagents] build '{n}' failed: {e}")
                continue
        out.append(_cache[n])
    return out
