"""
deep_agent — ДОПОЛНЕНИЕ к ядру (не замена): запускает LangChain DeepAgent для
долгогоризонтных/файловых подзадач. Даёт виртуальную ФС + todo-состояние + спавн
суб-агентов «из коробки», не трогая основной граф.

Вызывается из step_executor как обычный инструмент, когда подшаг тяжёлый и
многоступенчатый (исследовать → писать файлы → собирать результат). Для простых
шагов НЕ нужен — это дорогой путь.
"""
import os
from langchain_core.tools import tool

try:
    from deepagents import create_deep_agent
    from langchain_openai import ChatOpenAI
    from omegaconf import OmegaConf

    _cfg = OmegaConf.load("config.yml")
    _MODEL = _cfg.get("code_model", {}).get("name", "gpt-4o-mini")
    _DEEP = True
except Exception:  # noqa: BLE001
    _DEEP = False


@tool
def run_deep_agent(task: str, max_steps: int = 40) -> str:
    """
    Run a long-horizon DeepAgent (virtual filesystem + todo list + sub-agents) for a
    complex, multi-step task. Use ONLY for heavy tasks that need scratch files,
    planning across many steps, or delegating to sub-agents. Overkill for simple steps.

    Args:
        task: The complex task description for the deep agent.
        max_steps: Recursion limit for the deep agent run.

    Returns:
        The deep agent's final result (and a note about files it created).
    """
    if not _DEEP:
        return "DeepAgent недоступен (пакет deepagents не установлен)."
    key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return "Нет API-ключа для DeepAgent."
    try:
        model = ChatOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", model=_MODEL, temperature=0)
        agent = create_deep_agent(
            model=model,
            system_prompt=(
                "You are a focused deep agent for a long-horizon subtask. Use your todo list "
                "to plan, the virtual filesystem for scratch work, and sub-agents for isolated "
                "sub-steps. Finish with a concise final answer."
            ),
        )
        result = agent.invoke(
            {"messages": [("user", task)]},
            config={"recursion_limit": max_steps},
        )
        msgs = result.get("messages", [])
        final = msgs[-1].content if msgs else ""
        files = result.get("files", {}) or {}
        note = f"\n\n[Файлы во виртуальной ФС: {', '.join(files)}]" if files else ""
        return (final if isinstance(final, str) else str(final)) + note
    except Exception as e:  # noqa: BLE001
        return f"Ошибка DeepAgent: {type(e).__name__}: {e}"
