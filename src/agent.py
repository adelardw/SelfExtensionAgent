import re
import asyncio
import threading
import warnings
import os
from dotenv import load_dotenv
load_dotenv()

# Шумные warnings от langchain with_structured_output (pydantic-сериализация).
# Текст начинается с "Pydantic serializer warnings:" + перенос строки, поэтому
# матчим по началу (re.match), а не по PydanticSerializationUnexpectedValue (он за \n).
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from omegaconf import OmegaConf

from .schemas import GeneralGraphState
from .utils import (
    _format_chat_history,
    _run_smoke_test,
    _skill_loadable,
    ensure_python_package,
    missing_module_from_error,
)
from .prompts import (
    router_prompt,
    create_skills_system_prompt,
    sgr_create_prompt,
    test_case_prompt,
    skill_selector_prompt,
    validation_prompt,
    memory_extraction_prompt,
    reflection_prompt,
    step_execution_system_prompt,
    step_validation_prompt,
    synthesize_prompt,
    skill_retention_prompt,
    OPTIMIZABLE_PROMPTS,
)
from .structured_outputs import (
    RouteDecision,
    SkillSelection,
    SGRCreateResult,
    SkillTestCase,
    ValidationResult,
    MemoryExtraction,
    ReflectionResult,
    GoalAssessment,
    ReflexionDecision,
    TaskDecomposition,
    StepOutcome,
    IntegrationReview,
    SkillRetention,
    ClarificationSet,
)
from . import clarify
from .memory import MemoryStore, build_embedder, detect_implicit_feedback
from .improve import get_prompt as get_prompt_override, maybe_auto_improve
from .improve.prompt_store import format_fewshots, add_fewshot
from .external import get_external_context, format_external_context
from .mcp_client import suggest_server, get_mcp_tools, discover_mcp, approve_server
from .subagents import get_subagent_tools
from .tracing import traced, new_run, current_run, trace_store, diagnose
from .tools import get_manager_tools, get_all_loaded_skill_tools, get_skill_runtime_prompts, sync_registry
from .tools.skill_creation import (
    get_skills_for_prompt,
    read_skill,
    load_skill_tools,
    delete_skill,
    pop_last_created,
    mark_temporary,
    clear_temporary,
    _delete_skill_impl,
    _load_registry,
)


config = OmegaConf.load("config.yml")

MAX_CREATE_RETRIES: int = config.agent.max_create_retries
MAX_GLOBAL_RETRIES: int = config.agent.max_global_retries
LOW_CONF: float = config.agent.low_confidence_threshold
MAX_SUBTASK_RETRIES: int = config.agent.get("max_subtask_retries", 2)
MAX_SUBTASKS: int = config.agent.get("max_subtasks", 6)
AMBIGUITY_GATE: float = config.agent.get("ambiguity_gate", 0.6)
CLARIFY_SOFT_GATE: float = config.agent.get("clarify_soft_gate", 0.3)
CONSENSUS_VALIDATION: bool = config.agent.get("consensus_validation", True)
MAX_REVISIONS: int = config.agent.get("max_revisions", 1)
RETRY_CONF: float = config.agent.get("retry_confidence", 0.5)
STEP_ITER_LIMIT: int = config.agent.get("step_iter_limit", 16)

RECALL_K: int = config.get("memory", {}).get("recall_k", 5)
REFLECT_EVERY: int = config.get("memory", {}).get("reflect_every", 5)
RECALL_BUDGET: int = config.get("memory", {}).get("recall_budget_chars", 1800)
MEM_CAPS = dict(
    max_episodes=config.get("memory", {}).get("max_episodes", 2000),
    max_facts=config.get("memory", {}).get("max_facts", 300),
    max_reflections=config.get("memory", {}).get("max_reflections", 200),
)

memory_store = MemoryStore(
    db_path=config.get("memory", {}).get("db_path", "data/memory.db"),
    embedder=build_embedder(
        config.get("memory", {}).get("embeddings", False),
        config.get("memory", {}).get("embedding_model"),
    ),
)

# Автообновление реестра навыков при старте: подхватить orphan-скиллы,
# вычистить битые записи, проставить защиту базовым навыкам.
if config.get("skills", {}).get("autosync", True):
    _sync = sync_registry()
    if any(_sync.values()):
        print(f"[SkillRegistry] sync: {_sync}")

from .llm import chat as _chat


def rebuild_llms() -> None:
    """
    (Пере)создаёт LLM-клиентов и все цепочки по текущему провайдеру/модели.
    Ноды графа читают эти модульные глобалы при вызове, поэтому смена провайдера в
    рантайме (CLI /model) подхватывается без пересборки графа.
    """
    global llm, code_llm, deep_llm, route_chain, sgr_create_chain, test_case_chain
    global skill_selector_chain, validation_chain, validation_chain_b
    global memory_extraction_chain, reflection_chain, step_validation_chain
    global synth_chain, create_skills_agent, skill_retention_chain

    llm = _chat("fast", config.model.temperature)
    code_llm = _chat("code", config.code_model.temperature)
    # deep — редкие тяжёлые вызовы (heavy-ревью); дороже, поэтому только точечно.
    deep_llm = _chat("deep", config.get("deep_model", {}).get("temperature", 0))

    route_chain = router_prompt | llm.with_structured_output(RouteDecision)
    sgr_create_chain = sgr_create_prompt | llm.with_structured_output(SGRCreateResult)
    test_case_chain = test_case_prompt | llm.with_structured_output(SkillTestCase)
    skill_selector_chain = skill_selector_prompt | llm.with_structured_output(SkillSelection)
    validation_chain = validation_prompt | llm.with_structured_output(ValidationResult)
    validation_chain_b = validation_prompt | code_llm.with_structured_output(ValidationResult)  # консенсус
    memory_extraction_chain = memory_extraction_prompt | llm.with_structured_output(MemoryExtraction)
    reflection_chain = reflection_prompt | llm.with_structured_output(ReflectionResult)
    # Когнитивные ноды (goal/reflexion/decompose/fast_answer/reason) строятся через
    # _override_system → их промпты обучаемы (см. graph_learn).
    step_validation_chain = step_validation_prompt | llm.with_structured_output(StepOutcome)
    synth_chain = synthesize_prompt | llm
    skill_retention_chain = skill_retention_prompt | llm.with_structured_output(SkillRetention)

    create_skills_agent = create_agent(code_llm, get_manager_tools(), system_prompt=create_skills_system_prompt)


rebuild_llms()

def _override_system(role: str, sysvars: dict) -> str:
    """
    Системный промпт ноды = обучаемый параметр: берём override из ParamStore,
    иначе дефолт из OPTIMIZABLE_PROMPTS. Так backward может оптимизировать любую
    когнитивную ноду графа, не трогая исходники. Падение формата → дефолт.
    """
    default = OPTIMIZABLE_PROMPTS[role]
    tmpl = get_prompt_override(role, default)
    try:
        return tmpl.format(**sysvars)
    except (KeyError, IndexError, ValueError):
        return default.format(**sysvars)


async def _structured(role: str, schema, sysvars: dict, query: str):
    """Структурный вызов ноды через overrideable system + few-shots-агностично."""
    sys_text = _override_system(role, sysvars)
    return await llm.with_structured_output(schema).ainvoke(
        [SystemMessage(content=sys_text), HumanMessage(content=query)]
    )


async def recall_node(state: GeneralGraphState) -> dict:
    """
    Reflective-контур (вход): поднимает долгую память и формирует гипотезу
    неявной обратной связи ДО роутинга. Выполняется один раз на запрос
    (ретраи возвращаются в router, минуя recall).
    """
    new_run()  # старт нового трейс-прохода
    clarify.reset_ledger()  # чистый реестр уточнений на этот прогон
    user_id = state.get("user_id") or "default"
    query = state["query"]

    memory_context = memory_store.recall(user_id, query, k=RECALL_K, budget=RECALL_BUDGET)
    summary = memory_store.get_summary(user_id)
    if summary:
        memory_context = f"[Саммари сессии]\n{summary}\n\n{memory_context}"

    # Онбординг: первый контакт с пользователем (нет ни эпизодов, ни фактов) —
    # агент должен мягко представиться и начать строить профиль, а не молча отвечать.
    if memory_store.episode_count(user_id) == 0 and not memory_store.get_facts(user_id):
        memory_context = (
            "[ОНБОРДИНГ — первый контакт с этим пользователем]\n"
            "ГЛАВНОЕ: сначала ПОЛНОСТЬЮ и по существу выполни запрос пользователя — "
            "онбординг НИКОГДА не заменяет ответ. И только В КОНЦЕ ответа добавь 1–2 "
            "дружелюбные фразы: представься (персональный агент с памятью, навыками, "
            "доступом к устройству и веб-поиском) и спроси, как обращаться к пользователю. "
            "НЕ вываливай весь список возможностей.\n\n"
        ) + memory_context
    implicit_fb = detect_implicit_feedback(memory_store, user_id, query, LOW_CONF)
    ext = get_external_context(user_id).model_dump()

    return {
        "user_id": user_id,
        "memory_context": memory_context,
        "implicit_feedback": implicit_fb or "Сигналов нет.",
        "external_context": ext,
    }


async def goal_node(state: GeneralGraphState) -> dict:
    """
    Само-рефлексия цели: определяет цель запроса и держит «стоящую» цель в контексте.
    Отдельный гибкий модуль (не слит с reflexion) — удерживается через memory_context.
    """
    user_id = state.get("user_id") or "default"
    active = memory_store.get_active_goal(user_id)
    active_text = active["aim"] if active else "Нет активной цели."

    try:
        assess = await _structured("goal", GoalAssessment, {
            "active_goal": active_text,
            "chat_history": _format_chat_history(state),
        }, state["query"])
    except Exception as e:  # noqa: BLE001
        print(f"[Goal] assessment failed: {e}")
        return {"aim": "", "standing_goal": active_text if active else ""}

    if assess.completes_active and active:
        memory_store.close_active_goal(user_id)
        standing, rubric = "", []
    elif assess.is_standing and assess.standing_goal:
        memory_store.set_goal(user_id, assess.standing_goal, assess.success_criteria)
        standing = assess.standing_goal
        rubric = assess.success_criteria or memory_store.goal_criteria(memory_store.get_active_goal(user_id))
    elif active:
        standing = active["aim"]
        rubric = memory_store.goal_criteria(active)
    else:
        standing, rubric = "", []

    mem = state.get("memory_context", "") or ""
    if standing:
        block = f"🎯 ТЕКУЩАЯ ЦЕЛЬ (держи в контексте): {standing}"
        if rubric:
            block += "\nКритерии готовности:\n" + "\n".join(f"  ☐ {c}" for c in rubric)
        mem = f"{block}\n\n{mem}"

    return {"aim": assess.aim, "standing_goal": standing, "goal_rubric": rubric, "memory_context": mem}


async def reflexion_node(state: GeneralGraphState) -> dict:
    """
    Self-Reflexion Choice (мета-контроллер): по анализу задачи гибко выбирает режим
    взаимодействия — fast (System 1) / deliberate (System 2) / clarify. Отдельный
    модуль, чтобы режим можно было менять независимо от целеполагания.
    """
    try:
        decision = await _structured("reflexion", ReflexionDecision, {
            "memory_context": state.get("memory_context", "Память пуста."),
            "chat_history": _format_chat_history(state),
        }, state["query"])
    except Exception as e:  # noqa: BLE001
        print(f"[Reflexion] failed, fallback deliberate: {e}")
        return {"mode": "deliberate"}

    # Ambiguity-гейт (идея Ouroboros): слишком неоднозначно → переспросить, а не гадать.
    if decision.ambiguity >= AMBIGUITY_GATE and decision.mode != "clarify":
        mem = state.get("memory_context", "") or ""
        need = decision.missing_info or "уточни, что именно нужно"
        return {"mode": "clarify", "memory_context": f"⚠ Неясно (ambiguity {decision.ambiguity:.0%}): {need}\n\n{mem}"}
    # Средняя неоднозначность на путях с инструментами → не гадать молча, а собрать
    # батч уточнений ПЕРЕД исполнением (clarify_gate). Низкая — пропускаем (нулевая цена).
    soft = CLARIFY_SOFT_GATE <= decision.ambiguity < AMBIGUITY_GATE
    return {"mode": decision.mode, "needs_clarify_gate": soft}


async def fast_answer_node(state: GeneralGraphState) -> dict:
    """System 1: быстрый интуитивный ответ из памяти без инструментов (или уточняющий вопрос)."""
    sys_text = _override_system("fast_answer", {
        "memory_context": state.get("memory_context", "Память пуста."),
        "chat_history": _format_chat_history(state),
        "mode": state.get("mode", "fast"),
    })
    resp = await llm.ainvoke([SystemMessage(content=sys_text), HumanMessage(content=state["query"])])
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return {"final_answer": answer}


async def reason_node(state: GeneralGraphState) -> dict:
    """System 2 без инструментов: глубокое пошаговое рассуждение → продуманный ответ.
    Отдельный «тип мышления» в Any-2-Any; его промпт — обучаемый параметр (role 'reason')."""
    sys_text = _override_system("reason", {
        "memory_context": state.get("memory_context", "Память пуста."),
        "chat_history": _format_chat_history(state),
    })
    resp = await llm.ainvoke([SystemMessage(content=sys_text), HumanMessage(content=state["query"])])
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return {"final_answer": answer}


async def clarify_gate_node(state: GeneralGraphState) -> dict:
    """
    Онбординг неясной задачи ПЕРЕД исполнением: по контексту формирует батч точных
    вопросов (маркеры где набор конечен, открытые где нет), задаёт их человеку через
    зарегистрированный канал (REPL/бот) и кладёт ответы в общий ledger прогона.
    Нет ответа/канала → разумные допущения (status=assumed). Ответы переиспользуются
    всеми нодами ниже (decompose/step/synthesize) — агент не переспрашивает дважды.
    """
    try:
        result = await _structured("clarify_gate", ClarificationSet, {
            "memory_context": state.get("memory_context", "Память пуста."),
        }, state["query"])
    except Exception as e:  # noqa: BLE001
        print(f"[ClarifyGate] failed, пропускаю: {e}")
        return {}

    items = [{"question": it.question, "options": it.options, "assume": it.assume}
             for it in result.items[:4]]
    if not items:
        return {}
    resolved = await clarify.ask(items)  # канал человека или авто-допущения → ledger
    return {"clarifications": resolved}


async def router_node(state: GeneralGraphState) -> dict:
    """Решает: создать новый навык ИЛИ использовать существующие."""
    available = get_skills_for_prompt.invoke({})

    result = await route_chain.ainvoke({
        "query": state["query"],
        "available_skills": available or "Навыков пока нет.",
        "chat_history": _format_chat_history(state),
        "memory_context": state.get("memory_context", "Память пуста."),
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


    # Структурный канал: create_skill сам фиксирует имя при успехе (pop_last_created).
    # Регэксп по тексту сообщений — только фолбэк на случай нестандартного пути.
    created_name = pop_last_created()
    if not created_name:
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

    # ── Этап 1: статический ревью (LLM) ──
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

    # ── Этап 2: загружаемость (без LLM) — модуль импортируется и содержит @tool ──
    loadable, lmsg = _skill_loadable(skill_name)
    if not loadable:
        # Недостающая python-зависимость → ставим САМИ (uv add), не просим пользователя.
        mod = missing_module_from_error(lmsg)
        if mod:
            ok, note = await asyncio.to_thread(ensure_python_package, mod)
            print(f"[SGR] авто-зависимость '{mod}': {note}")
            if ok:
                loadable, lmsg = _skill_loadable(skill_name)
    if not loadable:
        delete_skill.invoke({"name": skill_name})
        return {
            "create_validation_passed": False,
            "create_feedback": (
                f"[Загрузка] Навык не импортируется или в нём нет рабочего @tool: {lmsg}. "
                f"Проверь, что есть 'from langchain_core.tools import tool', все импорты на месте "
                f"и нет синтаксических ошибок."
            ),
            "create_retries": retries + 1,
        }

    # ── Этап 3: runtime smoke-test (реальный вызов tool) ──
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
            # Smoke упал на недостающем модуле (ленивый import внутри функции) →
            # авто-установка и один повтор.
            mod = missing_module_from_error(output)
            if mod:
                ok, note = await asyncio.to_thread(ensure_python_package, mod)
                print(f"[SGR] авто-зависимость '{mod}': {note}")
                if ok:
                    success, output = _run_smoke_test(
                        skill_name, test_case.tool_name, test_case.test_input,
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
        mark_temporary(skill_name)  # создан под задачу; судьбу решит retention-судья в reflect
        return {
            "create_validation_passed": True,
            "create_feedback": "",
            "create_retries": retries,
        }

    except Exception as e:
        # Smoke-тест не сгенерился, НО этап 2 уже гарантировал загружаемость → принимаем.
        print(f"[SGR] Smoke test generation failed: {e}; навык загружаем (этап 2 пройден), принимаю.")
        load_skill_tools.invoke({"name": skill_name})
        mark_temporary(skill_name)
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


async def capability_research_node(state: GeneralGraphState) -> dict:
    """
    Capability-gap: если под задачу не нашлось навыка — НЕ строим сразу медленный
    навык, а сперва ищем в интернете «как это делается» и есть ли готовый MCP.
    Найденное кладём в capability_hint → исполнитель решает задачу общими
    инструментами по найденному способу. Создание навыка — крайняя мера.
    """
    selected = list(state.get("selected_skills", []))
    # Если навык под задачу выбран (в т.ч. device_control/app_control) — НЕ лезем в веб.
    if selected:
        return {"selected_skills": selected, "capability_gap": False, "capability_hint": "Навык под задачу есть."}

    # Реальный пробел (ничего не подошло) — только тогда подключаем поиск и ищем способ.
    selected = ["web_search"]
    gap = True

    # Реальный пробел: гуглим «как это делается» + есть ли готовый MCP, прежде чем что-то строить.
    query = state["query"]
    parts = []
    try:
        tools = {t.name: t for t in get_all_loaded_skill_tools(["web_search"])}
        search = tools.get("search_web")
        if search:
            how = search.invoke({"query": f"как сделать: {query}", "max_results": 4})
            mcp = search.invoke({"query": f"MCP server tool for {query[:60]}", "max_results": 3})
            parts.append("[Как это делается — из поиска]\n" + str(how)[:1200])
            parts.append("[Возможные готовые MCP/библиотеки]\n" + str(mcp)[:700])
    except Exception as e:  # noqa: BLE001
        parts.append(f"(поиск способа не удался: {e})")

    hint = "\n\n".join(parts) or "Готового способа не нашёл — собери из общих инструментов."

    # MCP: сначала доверенный каталог (авто-подключение), иначе discovery в реестре.
    mcp_servers: list[str] = []
    trusted = suggest_server(query)
    if trusted:
        mcp_servers = [trusted]
        hint = f"[Доверенный MCP: {trusted} — инструменты будут доступны]\n\n" + hint
    else:
        cand = discover_mcp(query, limit=config.get("mcp", {}).get("discover_limit", 8))
        if cand:
            auto = config.get("mcp", {}).get("auto_trust_discovered", False)
            if auto:
                top = cand[0]
                approve_server(top["name"], top["spec"])
                mcp_servers = [top["name"]]
                hint = f"[Найден и авто-подключён MCP: {top['name']} ({top['package']})]\n\n" + hint
            else:
                lst = "; ".join(f"{c['name']} ({c['package']})" for c in cand[:4])
                hint = (f"[Найдены MCP-серверы под задачу (нужно подтверждение пользователя для подключения): "
                        f"{lst}]\n\n" + hint)

    return {"selected_skills": selected, "capability_gap": True, "capability_hint": hint, "mcp_servers": mcp_servers}


async def decompose_node(state: GeneralGraphState) -> dict:
    """
    Декомпозиция в смешанном формате: ризонинг, внутри которого рождается план
    из атомарных подшагов (каждый со своим done_check). Заменяет линейный planning —
    дальше подшаги исполняются и валидируются по-пунктово.
    """
    selected = state.get("selected_skills", [])

    context_parts = [read_skill.invoke({"name": name}) for name in selected]
    skill_context = (
        "\n\n---\n\n".join(context_parts)
        if context_parts
        else "Навыки не выбраны — используй общие manager-инструменты."
    )

    rubric = state.get("goal_rubric", []) or []
    rubric_text = "\n".join(f"- {c}" for c in rubric) if rubric else "Rubric не задан."

    result = await _structured("decompose", TaskDecomposition, {
        "skill_context": skill_context,
        "memory_context": state.get("memory_context", "Память пуста."),
        "goal_rubric": rubric_text,
        "external_context": format_external_context(state.get("external_context")),
        "capability_hint": state.get("capability_hint", "Навык под задачу есть."),
        "clarifications": clarify.format_ledger(),
    }, state["query"])

    subtasks = [
        {"goal": st.goal, "done_check": st.done_check, "status": "pending", "result": ""}
        for st in result.subtasks[:MAX_SUBTASKS]
    ] or [{"goal": state["query"], "done_check": "Запрос пользователя выполнен.", "status": "pending", "result": ""}]

    plan_text = f"Подход: {result.reasoning}\n\nШаги:\n" + "\n".join(
        f"  {i}. {s['goal']} (готово, если: {s['done_check']})" for i, s in enumerate(subtasks, 1)
    )

    return {
        "skill_context": skill_context,
        "plan": plan_text,
        "subtasks": subtasks,
        "current_step": 0,
        "step_results": [],
        "step_retries": 0,
        "step_feedback": "",
    }



async def skill_injection_node(state: GeneralGraphState) -> dict:
    """Загружает tools выбранных навыков, собирает их промпты и фиксирует active_tools."""
    selected = state.get("selected_skills", [])

    for name in selected:
        try:
            load_skill_tools.invoke({"name": name})
        except Exception:
            pass

    skill_prompts = get_skill_runtime_prompts(selected)
    # Только инструменты выбранных навыков — без раздувания контекста всем реестром.
    active = [t.name for t in get_all_loaded_skill_tools(selected)]

    return {"skill_prompts": skill_prompts, "active_tools": active}



async def step_executor_node(state: GeneralGraphState) -> dict:
    """
    Исполняет ОДИН текущий подшаг и тут же валидирует его по done_check.
    Пройден → результат в step_results, переходим к следующему.
    Не пройден → ретрай этого же подшага с замечанием (до MAX_SUBTASK_RETRIES),
    после исчерпания — фиксируем как есть и идём дальше (мягкая деградация).
    """
    selected = state.get("selected_skills", [])
    # Анти-bloat: management-тулы (создание/удаление навыков) исполнителю шага НЕ нужны —
    # они живут в create_skills-ветке. Каждый лишний тул = плата за описание в КАЖДОМ
    # витке ReAct-цикла шага.
    tools = get_all_loaded_skill_tools(selected)

    # Автоподключение доверенных MCP-серверов, подобранных под задачу.
    mcp_names = []
    for mcp_tool in await get_mcp_tools(state.get("mcp_servers") or []):
        tools.append(mcp_tool)
        mcp_names.append(mcp_tool.name)

    # Под-агенты как базовые инструменты (agent-as-tool): доступны исполнителю наравне с навыками.
    subagent_names = []
    for sa in get_subagent_tools():
        tools.append(sa)
        subagent_names.append(sa.name)

    # Догон-уточнение: исполнитель может спросить пользователя, упёршись в развилку.
    tools.append(clarify.make_ask_user_tool())

    # Трейсим подключение инструментов (какие скиллы/MCP/под-агенты реально подцепились).
    try:
        trace_store.record(
            current_run(), "tools_attached", 0.0, "ok",
            output=f"skills={selected} mcp={mcp_names} subagents={subagent_names}",
        )
    except Exception:  # noqa: BLE001
        pass

    # Копия, НЕ мутируем state in-place (рассинхрон с чекпойнтерами LangGraph).
    subtasks = list(state.get("subtasks", []))
    idx = state.get("current_step", 0)
    step = subtasks[idx]
    prior = state.get("step_results", [])
    prior_text = "\n".join(f"{i+1}. {r['goal']} → {r['result'][:200]}" for i, r in enumerate(prior)) or "(нет)"

    # Берём оптимизированный self-learning'ом промпт шага + собранные few-shots.
    step_template = get_prompt_override("step_execution", step_execution_system_prompt)
    fmt = dict(
        memory_context=state.get("memory_context", "Память пуста."),
        external_context=format_external_context(state.get("external_context")),
        implicit_feedback=state.get("implicit_feedback", "Сигналов нет."),
        fewshots=format_fewshots("step_execution", k=3),
        capability_hint=state.get("capability_hint", "Навык под задачу есть."),
        clarifications=clarify.format_ledger(),
        prior_steps=prior_text,
        step_goal=step["goal"],
        step_done_check=step["done_check"],
        step_feedback=state.get("step_feedback", "") or "(первая попытка)",
    )
    try:
        system = step_template.format(**fmt)
    except (KeyError, IndexError):
        system = step_execution_system_prompt.format(**fmt)

    agent = create_agent(code_llm, tools, system_prompt=system)
    # Лимит витков ReAct-цикла шага: каждый виток ресендит весь контекст + растущие
    # выводы тулов — без лимита один шаг может «разогнаться» на десятки тысяч токенов.
    try:
        result = await agent.ainvoke(
            {"messages": [("human", step["goal"])]},
            config={"recursion_limit": STEP_ITER_LIMIT},
        )
        last = result["messages"][-1]
        output = last.content if hasattr(last, "content") else str(last)
    except Exception as e:  # noqa: BLE001 — GraphRecursionError и пр.: мягкая деградация шага
        output = f"(шаг прерван: {type(e).__name__} — вероятно, превышен лимит итераций {STEP_ITER_LIMIT})"

    # По-пунктовая валидация
    try:
        outcome = await step_validation_chain.ainvoke({
            "step_goal": step["goal"],
            "step_done_check": step["done_check"],
            "step_output": output,
        })
        passed, note = outcome.passed, outcome.note
    except Exception as e:  # noqa: BLE001
        passed, note = True, f"(валидация шага пропущена: {e})"

    retries = state.get("step_retries", 0)
    if not passed and retries < MAX_SUBTASK_RETRIES:
        return {"step_retries": retries + 1, "step_feedback": note}

    # принимаем шаг (пройден или исчерпали ретраи) и двигаемся дальше
    subtasks[idx] = {**step, "status": "done" if passed else "partial", "result": output}
    new_results = prior + [{"goal": step["goal"], "result": output, "passed": passed}]
    return {
        "subtasks": subtasks,
        "step_results": new_results,
        "current_step": idx + 1,
        "step_retries": 0,
        "step_feedback": "",
        "active_mcp_tools": mcp_names,
    }


async def synthesize_node(state: GeneralGraphState) -> dict:
    """Собирает финальный ответ из результатов всех подшагов."""
    results = state.get("step_results", [])
    if len(results) == 1:
        # одношаговая задача — результат и есть ответ, без лишнего LLM-вызова
        return {"final_answer": results[0]["result"]}

    results_text = "\n\n".join(
        f"[Шаг {i+1}] {r['goal']}\n{r['result']}" for i, r in enumerate(results)
    ) or "(шаги не дали результата)"

    resp = await synth_chain.ainvoke({
        "query": state["query"],
        "memory_context": state.get("memory_context", "Память пуста."),
        "clarifications": clarify.format_ledger(),
        "step_results": results_text,
    })
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return {"final_answer": answer}


async def review_node(state: GeneralGraphState) -> dict:
    """
    Heavy-режим: сквозной ревью СОБРАННОГО решения (deep-модель, редкий дорогой вызов).
    Человеческий паттерн «большая работа → перечитать целиком перед сдачей»: смотрит на
    полный пайплайн запрос→шаги→финал, находит интеграционные проблемы и добавляет
    подшаги доработки обратно в цикл step_executor (до MAX_REVISIONS раундов).
    """
    rounds = state.get("revision_rounds", 0)
    steps_text = "\n".join(
        f"[Шаг {i+1}] {r['goal']} → {r['result'][:400]}" for i, r in enumerate(state.get("step_results", []))
    ) or "(шагов не было)"
    rubric = state.get("goal_rubric", []) or []
    sys_text = _override_system("review", {
        "goal_rubric": "\n".join(f"- {c}" for c in rubric) if rubric else "(rubric не задан)",
    })
    try:
        review = await deep_llm.with_structured_output(IntegrationReview).ainvoke([
            SystemMessage(content=sys_text),
            HumanMessage(content=(
                f"Исходная задача: {state['query']}\n\n"
                f"Выполненные шаги:\n{steps_text}\n\n"
                f"Собранный финальный ответ:\n{state.get('final_answer', '')[:4000]}"
            )),
        ])
    except Exception as e:  # noqa: BLE001
        print(f"[Review] failed, пропускаю ревью: {e}")
        return {"revision_rounds": rounds + 1}

    if review.passed or not review.fix_subtasks:
        return {"revision_rounds": rounds + 1}

    # Доработка: добавляем fix-подшаги в конец плана и возвращаемся в шаговый цикл.
    fixes = [
        {"goal": st.goal, "done_check": st.done_check, "status": "pending", "result": ""}
        for st in review.fix_subtasks[:3]
    ]
    problems = "; ".join(review.problems[:5])
    print(f"[Review] найдены проблемы: {problems} → {len(fixes)} fix-подшагов")
    return {
        "subtasks": list(state.get("subtasks", [])) + fixes,
        "revision_rounds": rounds + 1,
        "step_feedback": f"Сквозной ревью нашёл проблемы: {problems}",
    }


async def validation_node(state: GeneralGraphState) -> dict:
    """Финальный Schema Guided Reasoning ответа агента."""
    rubric = state.get("goal_rubric", []) or []
    rubric_text = "\n".join(f"- {c}" for c in rubric) if rubric else "Rubric не задан — оцени по общим критериям."

    payload = {
        "query": state["query"],
        "final_answer": state.get("final_answer", "Ответ не сгенерирован."),
        "chat_history": _format_chat_history(state),
        "goal_rubric": rubric_text,
    }
    result = await validation_chain.ainvoke(payload)
    is_valid, confidence, feedback = result.is_valid, result.confidence, result.feedback

    # Мульти-модельный консенсус (идея Ouroboros): второй судья на другой модели.
    if CONSENSUS_VALIDATION:
        try:
            b = await validation_chain_b.ainvoke(payload)
            agree = (b.is_valid == result.is_valid)
            is_valid = result.is_valid and b.is_valid
            # согласие → берём min уверенности; разногласие → штраф (двигает на ретрай).
            confidence = min(result.confidence, b.confidence) * (1.0 if agree else 0.6)
            if not agree:
                feedback = f"[консенсус: расхождение судей] {feedback} | 2-й: {b.feedback}"
        except Exception as e:  # noqa: BLE001
            print(f"[Validation] consensus skipped: {e}")

    retries = state.get("global_retries", 0)
    # Инкремент только когда реально уходим на полный ретрай (см. route_after_validation).
    bumped = retries + 1 if (not is_valid or confidence < RETRY_CONF) else retries

    return {
        "validation_passed": is_valid,
        "confidence": confidence,
        "validation_feedback": feedback,
        "global_retries": bumped,
    }

async def reflect_node(state: GeneralGraphState) -> dict:
    """
    Reflective-контур (выход): пишет эпизод в долгую память (trajectory-store),
    извлекает устойчивые факты о пользователе и периодически синтезирует выводы.
    Терминальная нода — выполняется только на успешном завершении (не на ретраях).
    """
    user_id = state.get("user_id") or "default"
    query = state["query"]
    answer = state.get("final_answer", "")
    confidence = state.get("confidence", 0.0)
    mode = state.get("mode", "deliberate")
    # Валидируемые режимы (deliberate, reason) считаются неудачей при низкой уверенности;
    # быстрые/уточняющие (fast, clarify) не валидируются → всегда ok, в тренд не идут.
    validated = mode in ("deliberate", "reason", "heavy")
    outcome = "ok" if (not validated or confidence >= LOW_CONF) else "low_conf"

    # 1. Эпизодическая память / trajectory-store
    ep_id = memory_store.add_episode(
        user_id=user_id,
        query=query,
        answer=answer,
        route=state.get("route", ""),
        skills=state.get("selected_skills", []),
        confidence=confidence,
        outcome=outcome,
        feedback=state.get("validation_feedback", ""),
        run_id=current_run(),
        mode=mode,
    )

    # Forward-харвест: успешный обдуманный прогон → few-shot пример (генерализация, без LLM).
    if mode in ("deliberate", "heavy") and confidence >= LOW_CONF and answer:
        try:
            add_fewshot("step_execution", query, answer, confidence)
        except Exception:  # noqa: BLE001
            pass

    # Судьба ВРЕМЕННОГО навыка, созданного по ходу задачи: оставить в библиотеке
    # (переиспользуем) или выбросить (одноразовый). Решается в фоне, дёшево.
    created_skill = state.get("created_skill_name", "")

    async def _judge_created_skill():
        if not created_skill:
            return
        meta = _load_registry().get(created_skill)
        if not meta or not meta.get("temporary"):
            return
        if outcome != "ok":
            return  # задача не решена — навык остаётся temp, TTL-чистка приберёт
        try:
            verdict = await skill_retention_chain.ainvoke({
                "query": query,
                "skill_name": created_skill,
                "skill_description": meta.get("description", ""),
                "confidence": f"{confidence:.0%}",
            })
            if verdict.keep:
                clear_temporary(created_skill)
                print(f"[SkillRetention] '{created_skill}' принят в библиотеку: {verdict.reason}")
            else:
                _delete_skill_impl(created_skill, allow_protected=False)
                print(f"[SkillRetention] '{created_skill}' одноразовый, удалён: {verdict.reason}")
        except Exception as e:  # noqa: BLE001
            print(f"[SkillRetention] judge failed (навык остаётся temp): {e}")

    # 2+3. Извлечение фактов и периодическая рефлексия — независимы → параллельно.
    async def _extract_facts():
        known = memory_store.get_facts(user_id)
        known_str = "\n".join(f"- {f['key']}: {f['value']}" for f in known) or "Пока ничего."
        extraction = await memory_extraction_chain.ainvoke({
            "known_facts": known_str,
            "query": query,
            "final_answer": answer,
        })
        for fact in extraction.facts:
            fid = memory_store.add_fact(
                user_id=user_id, key=fact.key, value=fact.value,
                importance=fact.importance, source_episode=ep_id,
                tags=getattr(fact, "tags", []),
            )
            # связываем факт с эпизодом-источником (граф памяти)
            memory_store.add_edge(user_id, "episode", ep_id, "fact", fid, "derived")

    async def _synth_reflection():
        if memory_store.episode_count(user_id) % REFLECT_EVERY != 0:
            return
        recent = memory_store.recent_episodes(user_id, n=REFLECT_EVERY)
        recent_str = "\n".join(f"- «{ep['query'][:80]}» → {ep['outcome']}" for ep in recent)
        existing = memory_store._conn.execute(  # noqa: SLF001
            "SELECT insight FROM reflections WHERE user_id=? ORDER BY ts DESC LIMIT 5",
            (user_id,),
        ).fetchall()
        existing_str = "\n".join(f"- {r['insight']}" for r in existing) or "Пока нет."
        reflection = await reflection_chain.ainvoke({
            "recent_episodes": recent_str,
            "existing_reflections": existing_str,
        })
        for insight in reflection.insights:
            memory_store.add_reflection(user_id, insight)
        # Обновляем локальное саммари сессии (SummaryCtx) — раз в REFLECT_EVERY, ради бюджета.
        memory_store.set_summary(
            user_id,
            "Недавняя активность: " + "; ".join(ep["query"][:60] for ep in recent[:5]),
        )

    # Тяжёлую пост-обработку (LLM-извлечение фактов, рефлексия, обслуживание, само-улучшение)
    # уносим в ФОН: ответ пользователю уже готов — не заставляем ждать ещё LLM-вызовы.
    def _post_reflect() -> None:
        async def _run():
            await asyncio.gather(_extract_facts(), _synth_reflection(), _judge_created_skill(),
                                 return_exceptions=True)
        try:
            asyncio.run(_run())
        except Exception as e:  # noqa: BLE001
            print(f"[Reflect-bg] extract/reflect failed: {e}")
        trend = memory_store.quality_trend(user_id)
        if trend["trend"] == "declining":
            print(f"[QualityMonitor] ⚠ деградация качества: {trend}")
        if memory_store.episode_count(user_id) % REFLECT_EVERY == 0:
            try:
                memory_store.prune(**MEM_CAPS)
                trace_store.prune()
                report = diagnose(memory_store, user_id)
                if not report["healthy"]:
                    print(f"[SelfDiagnosis] косяки: {report['findings']}")
            except Exception as e:  # noqa: BLE001
                print(f"[Reflect-bg] maintenance failed: {e}")
        try:
            maybe_auto_improve(memory_store, degrading=(trend["trend"] == "declining"))
        except Exception as e:  # noqa: BLE001
            print(f"[Reflect-bg] auto-improve failed: {e}")

    threading.Thread(target=_post_reflect, daemon=True).start()
    return {}


def route_after_reflexion(state: GeneralGraphState) -> str:
    """Meta-controller: fast/clarify → сразу ответчик (БЕЗ целеполагания, экономия вызова);
    reason/deliberate → сначала goal (целеполагание нужно для rubric/декомпозиции)."""
    if state.get("mode") in ("fast", "clarify"):
        return "fast_answer"
    return "goal"


def route_after_goal(state: GeneralGraphState) -> str:
    """После целеполагания: reason → рассуждение; средняя неоднозначность на пути с
    инструментами → clarify_gate (батч уточнений); иначе → router."""
    if state.get("mode") == "reason":
        return "reason"
    if state.get("needs_clarify_gate") and state.get("mode") in ("deliberate", "heavy"):
        return "clarify_gate"
    return "router"


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


def route_after_step(state: GeneralGraphState) -> str:
    """Шаговый цикл: остались подшаги (или идёт ретрай) → step_executor, иначе → synthesize."""
    if state.get("current_step", 0) < len(state.get("subtasks", [])):
        return "step_executor"
    return "synthesize"


def route_after_synthesize(state: GeneralGraphState) -> str:
    """Heavy-режим: после сборки решения — сквозной ревью (пока есть бюджет раундов);
    остальные режимы идут сразу на финальную валидацию."""
    if state.get("mode") == "heavy" and state.get("revision_rounds", 0) < MAX_REVISIONS:
        return "review"
    return "validation"


def route_after_review(state: GeneralGraphState) -> str:
    """Ревью добавил fix-подшаги → обратно в шаговый цикл; чисто → финальная валидация."""
    if state.get("current_step", 0) < len(state.get("subtasks", [])):
        return "step_executor"
    return "validation"


def route_after_validation(state: GeneralGraphState) -> str:
    """
    Final SGR → reflect | router (полный ретрай).
    Полный повтор пайплайна — САМАЯ дорогая операция (×2 токенов на запрос),
    поэтому ретраим только реально невалидный ответ (is_valid=False или
    confidence < retry_confidence). «Серединная» уверенность (0.5–0.7) —
    принимаем с замечанием: это пища для self-learning, а не повод сжечь бюджет.
    """
    invalid = not state.get("validation_passed", True)
    conf = state.get("confidence", 1.0)
    if (invalid or conf < RETRY_CONF) and state.get("global_retries", 0) < MAX_GLOBAL_RETRIES:
        return "router"
    return "reflect"


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """Собирает и компилирует граф. Checkpointer передаётся снаружи."""
    graph = StateGraph(GeneralGraphState)

    # Все ноды оборачиваются трейсером (имя, время, статус → data/traces.db).
    _nodes = {
        "recall": recall_node,
        "goal": goal_node,
        "reflexion": reflexion_node,
        "fast_answer": fast_answer_node,
        "reason": reason_node,
        "clarify_gate": clarify_gate_node,
        "router": router_node,
        "create_skills": create_skills_node,
        "sgr_create": sgr_create_node,
        "skill_selector": skill_selector_node,
        "capability_research": capability_research_node,
        "decompose": decompose_node,
        "skill_injection": skill_injection_node,
        "step_executor": step_executor_node,
        "synthesize": synthesize_node,
        "review": review_node,
        "validation": validation_node,
        "reflect": reflect_node,
    }
    for _name, _fn in _nodes.items():
        graph.add_node(_name, traced(_name, _fn))

    graph.add_edge(START, "recall")
    graph.add_edge("recall", "reflexion")   # сначала выбор режима (дёшево)
    graph.add_conditional_edges("reflexion", route_after_reflexion, {
        "fast_answer": "fast_answer",
        "goal":        "goal",
    })
    graph.add_conditional_edges("goal", route_after_goal, {
        "reason": "reason",
        "clarify_gate": "clarify_gate",
        "router": "router",
    })
    graph.add_edge("clarify_gate", "router")
    graph.add_edge("fast_answer", "reflect")
    graph.add_edge("reason", "validation")  # глубокое рассуждение проходит финальную валидацию
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

    graph.add_edge("skill_selector",  "capability_research")
    graph.add_edge("capability_research", "decompose")
    graph.add_edge("decompose", "skill_injection")
    graph.add_edge("skill_injection", "step_executor")
    graph.add_conditional_edges("step_executor", route_after_step, {
        "step_executor": "step_executor",  # ретрай шага / следующий шаг
        "synthesize":    "synthesize",
    })
    graph.add_conditional_edges("synthesize", route_after_synthesize, {
        "review":     "review",      # heavy: сквозной ревью собранного решения
        "validation": "validation",
    })
    graph.add_conditional_edges("review", route_after_review, {
        "step_executor": "step_executor",  # доработка fix-подшагов
        "validation":    "validation",
    })

    graph.add_conditional_edges("validation", route_after_validation, {
        "router":  "router",
        "reflect": "reflect",
    })
    graph.add_edge("reflect", END)

    return graph.compile(checkpointer=checkpointer)
