from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class GeneralGraphState(TypedDict):
    # ── Input ──
    query: str
    user_id: str  # идентификатор пользователя для долгой памяти (thread_id / tg user id)
    messages: Annotated[list, add_messages]
    chat_history: list[dict]  # [{role: "user"|"assistant", content: str}] — управляется из main.py

    # ── Memory (recall) ──
    memory_context: str       # инъектируемый блок долгой памяти
    implicit_feedback: str    # гипотеза о неявной обратной связи пользователя

    # ── Self-reflection (goal) ──
    aim: str                  # цель текущего запроса
    standing_goal: str        # активная «стоящая» цель, удерживаемая в контексте
    goal_rubric: list[str]    # критерии успеха активной цели (DeepAgents-style rubric)

    # ── Self-Reflexion Choice (meta-controller) ──
    mode: str                 # "fast" | "reason" | "deliberate" | "heavy" | "clarify" — тип мышления
    revision_rounds: int      # heavy: сколько раундов «сквозной ревью → доработка» уже прошло

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
