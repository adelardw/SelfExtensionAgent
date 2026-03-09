from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class GeneralGraphState(TypedDict):
    # ── Input ──
    query: str
    messages: Annotated[list, add_messages]
    chat_history: list[dict]  # [{role: "user"|"assistant", content: str}] — управляется из main.py

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
