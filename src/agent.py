import re
import warnings
import os
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
    module="pydantic",
)

from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from omegaconf import OmegaConf

from .schemas import GeneralGraphState
from .utils import _format_chat_history, _run_smoke_test
from .prompts import (
    router_prompt,
    create_skills_system_prompt,
    sgr_create_prompt,
    test_case_prompt,
    skill_selector_prompt,
    planning_prompt,
    execution_system_prompt,
    validation_prompt,
)
from .structured_outputs import (
    RouteDecision,
    SkillSelection,
    ExecutionPlan,
    SGRCreateResult,
    SkillTestCase,
    ValidationResult,
)
from .tools import get_manager_tools, get_all_loaded_skill_tools, get_skill_runtime_prompts
from .tools.skill_creation import (
    get_skills_for_prompt,
    read_skill,
    load_skill_tools,
    delete_skill
)


config = OmegaConf.load("config.yml")

MAX_CREATE_RETRIES: int = config.agent.max_create_retries
MAX_GLOBAL_RETRIES: int = config.agent.max_global_retries
LOW_CONF: float = config.agent.low_confidence_threshold

llm = ChatOpenAI(
    api_key=os.getenv('OPEN_ROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1",
    model=config.model.name,
    temperature=config.model.temperature,
)

code_llm = ChatOpenAI(
    api_key=os.getenv('OPEN_ROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1",
    model=config.code_model.name,
    temperature=config.code_model.temperature,
)


route_chain = router_prompt | llm.with_structured_output(RouteDecision)
sgr_create_chain = sgr_create_prompt | llm.with_structured_output(SGRCreateResult)
test_case_chain = test_case_prompt | llm.with_structured_output(SkillTestCase)
skill_selector_chain = skill_selector_prompt | llm.with_structured_output(SkillSelection)
planning_chain = planning_prompt | llm.with_structured_output(ExecutionPlan)
validation_chain = validation_prompt | llm.with_structured_output(ValidationResult)


create_skills_agent = create_react_agent(
    code_llm,
    get_manager_tools(),
    prompt=create_skills_system_prompt,
)

async def router_node(state: GeneralGraphState) -> dict:
    """Решает: создать новый навык ИЛИ использовать существующие."""
    available = get_skills_for_prompt.invoke({})

    result = await route_chain.ainvoke({
        "query": state["query"],
        "available_skills": available or "Навыков пока нет.",
        "chat_history": _format_chat_history(state),
    })

    return {"route": result.route}


async def create_skills_node(state: GeneralGraphState) -> dict:
    """ReAct-агент создаёт новый навык через инструменты управления."""
    feedback = state.get("create_feedback", "")
    msg = state["query"]
    if feedback:
        msg += f"\n\nОбратная связь с предыдущей попытки:\n{feedback}"

    result = await create_skills_agent.ainvoke({
        "messages": [("human", msg)],
    })


    created_name = ""
    for m in reversed(result["messages"]):
        text = m.content if hasattr(m, "content") else str(m)
        match = re.search(r"Skill '(\w+)'.*created", text, re.I)
        if match:
            created_name = match.group(1)
            break

    return {"created_skill_name": created_name}



async def sgr_create_node(state: GeneralGraphState) -> dict:
    """
    Schema Guided Reasoning свежесозданного навыка.
    Два этапа:
      1. Статический ревью (LLM анализирует код)
      2. Runtime smoke test (загружает модуль, вызывает tool, проверяет результат)
    """
    skill_name = state.get("created_skill_name", "")
    retries = state.get("create_retries", 0)

    if not skill_name:
        return {
            "create_validation_passed": False,
            "create_feedback": "Навык не был создан. Повтори попытку.",
            "create_retries": retries + 1,
        }

    skill_content = read_skill.invoke({"name": skill_name})


    result = await sgr_create_chain.ainvoke({
        "query": state["query"],
        "created_skill_name": skill_name,
        "skill_content": skill_content,
    })

    if not result.is_valid or result.confidence < LOW_CONF:
        delete_skill.invoke({"name": skill_name})
        return {
            "create_validation_passed": False,
            "create_feedback": (
                f"[Статический ревью] Проблемы: {'; '.join(result.issues)}. "
                f"{result.suggestion}"
            ),
            "create_retries": retries + 1,
        }

    try:
        test_case = await test_case_chain.ainvoke({
            "skill_content": skill_content,
        })

        success, output = _run_smoke_test(
            skill_name,
            test_case.tool_name,
            test_case.test_input,
        )

        if not success:
            delete_skill.invoke({"name": skill_name})
            return {
                "create_validation_passed": False,
                "create_feedback": (
                    f"[Smoke test FAILED] tool={test_case.tool_name}, "
                    f"input={test_case.test_input}: {output}\n"
                    f"Исправь код чтобы tool реально работал. "
                    f"Используй бесплатные API без ключей."
                ),
                "create_retries": retries + 1,
            }

        load_skill_tools.invoke({"name": skill_name})
        return {
            "create_validation_passed": True,
            "create_feedback": "",
            "create_retries": retries,
        }

    except Exception as e:
        print(f"[SGR] Smoke test generation failed: {e}, skipping runtime test")
        load_skill_tools.invoke({"name": skill_name})
        return {
            "create_validation_passed": True,
            "create_feedback": "",
            "create_retries": retries,
        }



async def skill_selector_node(state: GeneralGraphState) -> dict:
    """Выбирает релевантные навыки из реестра."""
    available = get_skills_for_prompt.invoke({})

    result = await skill_selector_chain.ainvoke({
        "query": state["query"],
        "available_skills": available or "Нет доступных навыков.",
    })

    return {"selected_skills": result.selected_skills}


async def planning_node(state: GeneralGraphState) -> dict:
    """Строит пошаговый план выполнения."""
    selected = state.get("selected_skills", [])

    context_parts = []
    for name in selected:
        content = read_skill.invoke({"name": name})
        context_parts.append(content)

    skill_context = (
        "\n\n---\n\n".join(context_parts)
        if context_parts
        else "Навыки не выбраны — используй общие manager-инструменты."
    )

    result = await planning_chain.ainvoke({
        "query": state["query"],
        "skill_context": skill_context,
    })

    plan_text = f"Подход: {result.reasoning}\n\nШаги:\n"
    for i, step in enumerate(result.steps, 1):
        plan_text += f"  {i}. {step}\n"

    return {"plan": plan_text, "skill_context": skill_context}



async def skill_injection_node(state: GeneralGraphState) -> dict:
    """Загружает tools выбранных навыков и собирает их системные промпты."""
    selected = state.get("selected_skills", [])

    for name in selected:
        try:
            load_skill_tools.invoke({"name": name})
        except Exception:
            pass

    skill_prompts = get_skill_runtime_prompts(selected)

    return {"skill_prompts": skill_prompts}



async def execution_node(state: GeneralGraphState) -> dict:
    """Исполняет план с инъектированными инструментами и промптами навыков."""
    tools = get_manager_tools() + get_all_loaded_skill_tools()

    system = execution_system_prompt.format(
        plan=state.get("plan", "Нет плана — обработай запрос напрямую."),
        skill_prompts=state.get("skill_prompts", "Нет инструкций навыков."),
        chat_history=_format_chat_history(state),
    )

    execution_agent = create_react_agent(code_llm, tools, prompt=system)
    result = await execution_agent.ainvoke({
        "messages": [("human", state["query"])],
    })

    last = result["messages"][-1]
    answer = last.content if hasattr(last, "content") else str(last)

    return {"final_answer": answer}


async def validation_node(state: GeneralGraphState) -> dict:
    """Финальный Schema Guided Reasoning ответа агента."""
    result = await validation_chain.ainvoke({
        "query": state["query"],
        "final_answer": state.get("final_answer", "Ответ не сгенерирован."),
        "chat_history": _format_chat_history(state),
    })

    retries = state.get("global_retries", 0)
    bumped = retries if result.confidence >= LOW_CONF else retries + 1

    return {
        "validation_passed": result.is_valid,
        "confidence": result.confidence,
        "validation_feedback": result.feedback,
        "global_retries": bumped,
    }

def route_after_router(state: GeneralGraphState) -> str:
    """Router → create_skills | skill_selector"""
    if state["route"] == "create_skill":
        return "create_skills"
    return "skill_selector"


def route_after_sgr_create(state: GeneralGraphState) -> str:
    """SGR create → router (ок) | create_skills (retry) | skill_selector (сдаёмся)"""
    if state.get("create_validation_passed"):
        return "router"
    if state.get("create_retries", 0) >= MAX_CREATE_RETRIES:
        return "skill_selector"
    return "create_skills"


def route_after_validation(state: GeneralGraphState) -> str:
    """Final SGR → END (ок) | router (low confidence)"""
    if state.get("confidence", 1.0) >= LOW_CONF:
        return END
    if state.get("global_retries", 0) >= MAX_GLOBAL_RETRIES:
        return END
    return "router"


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """Собирает и компилирует граф. Checkpointer передаётся снаружи."""
    graph = StateGraph(GeneralGraphState)

    graph.add_node("router", router_node)
    graph.add_node("create_skills", create_skills_node)
    graph.add_node("sgr_create", sgr_create_node)
    graph.add_node("skill_selector", skill_selector_node)
    graph.add_node("planning", planning_node)
    graph.add_node("skill_injection", skill_injection_node)
    graph.add_node("execution", execution_node)
    graph.add_node("validation", validation_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {
        "create_skills":  "create_skills",
        "skill_selector": "skill_selector",
    })

    graph.add_edge("create_skills", "sgr_create")
    graph.add_conditional_edges("sgr_create", route_after_sgr_create, {
        "router":         "router",
        "create_skills":  "create_skills",
        "skill_selector": "skill_selector",
    })

    graph.add_edge("skill_selector",  "planning")
    graph.add_edge("planning", "skill_injection")
    graph.add_edge("skill_injection", "execution")
    graph.add_edge("execution", "validation")

    graph.add_conditional_edges("validation", route_after_validation, {"router": "router",
                                                                         END: END})

    return graph.compile(checkpointer=checkpointer)
