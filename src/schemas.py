from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class GeneralGraphState(TypedDict):
    # ── Input ──
    query: str
    user_id: str  # идентификатор пользователя для долгой памяти (thread_id / tg user id)
    session_id: str  # идентификатор СЕССИИ/чата для временных приложенных файлов (ярус 3)
    messages: Annotated[list, add_messages]
    chat_history: list[dict]  # [{role: "user"|"assistant", content: str}] — управляется из main.py

    # ── Memory (recall) ──
    memory_context: str       # инъектируемый блок долгой памяти
    own_docs: bool            # AutoRAG нашёл СОБСТВЕННЫЕ документы юзера (БЗ/сессия) → reflexion глушит мнимый clarify
    implicit_feedback: str    # гипотеза о неявной обратной связи пользователя
    query_emb: list           # эмбеддинг запроса, посчитанный в recall ОДИН раз — переиспользуется (intent-роутер, без лишних вызовов)

    # ── Self-reflection (goal) ──
    aim: str                  # цель текущего запроса
    standing_goal: str        # активная «стоящая» цель, удерживаемая в контексте
    goal_rubric: list[str]    # критерии успеха активной цели (DeepAgents-style rubric)

    # ── Self-Reflexion Choice (meta-controller) ──
    mode: str                 # "fast" | "reason" | "act" | "deliberate" | "heavy" | "clarify" — тип мышления
    force_mode: str           # юзер зафиксировал режим в /config — reflexion не выбирает ("" = авто)
    revision_rounds: int      # heavy: сколько раундов «сквозной ревью → доработка» уже прошло
    steps_executed: int       # глобальный счётчик исполнений шага на прогон (бюджет от runaway)
    needs_clarify_gate: bool  # средняя неоднозначность → собрать батч уточнений перед исполнением
    clarifications: list[dict]  # реестр уточнений прогона (вопрос/варианты/ответ/статус)
    user_blocked: bool        # действие отклонено пользователем (HITL) — не ретраить, не винить агента

    # ── Capabilities ──
    active_tools: list[str]      # имена локальных инструментов навыков в execution
    active_mcp_tools: list[str]  # имена инструментов, реально подключённых из MCP-серверов
    mcp_servers: list[str]       # доверенные MCP-серверы, подобранные под задачу (для подключения)
    capability_gap: bool         # агент осознал нехватку экспертизы → нужен поиск/MCP
    capability_hint: str         # найденный в интернете способ «как это делается» / варианты MCP

    # ── External agent (A2A / MCP) ──
    external_context: dict    # с кем взаимодействуем извне и что он умеет

    # ── Task decomposition + per-step execution ──
    subtasks: list[dict]      # пункты плана: [{goal, done_check, status, result}]
    current_step: int         # индекс текущего подшага
    step_results: list[dict]  # накопленные результаты выполненных подшагов
    step_retries: int         # ретраи текущего подшага
    step_feedback: str        # замечание валидатора по проваленному подшагу

    # ── Router ──
    route: str  # "create_skill" | "use_skills"

    # ── Create Skills branch ──
    created_skill_name: str
    create_validation_passed: bool
    create_feedback: str
    create_retries: int

    # ── Use Skills branch ──
    recipe_id: int            # применённый рецепт (скомпилированный опыт); 0 — холодный прогон
    selected_skills: list[str]
    plan: str
    skill_context: str
    skill_prompts: str  # инъектированные системные промпты навыков

    # ── Execution ──
    final_answer: str

    # ── Final Validation (SGR) ──
    confidence: float
    validation_passed: bool
    validation_feedback: str
    global_retries: int
    web_research_used: bool  # был дисциплинированный agentic research (факты уже верифицированы)
