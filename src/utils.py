from .tools.skill_creation import SKILLS_DIR
from .schemas import GeneralGraphState
import importlib
import sys
import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

SMOKE_TEST_TIMEOUT: int = 15

def _run_smoke_test(skill_name: str, tool_name: str, test_input: dict) -> tuple[bool, str]:
    """
    Загружает модуль навыка, находит @tool функцию, вызывает с тестовым вводом.
    Возвращает (success, result_or_error).
    """
    py_file = SKILLS_DIR / skill_name / f"{skill_name}.py"
    if not py_file.exists():
        return False, f"Файл {py_file} не найден"

    try:

        module_name = f"skills_test.{skill_name}"
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, str(py_file))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        return False, f"Ошибка загрузки модуля: {type(e).__name__}: {e}"

    tool_func = None
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if hasattr(obj, "name") and hasattr(obj, "invoke"):
            if obj.name == tool_name or attr_name == tool_name:
                tool_func = obj
                break

    if tool_func is None:
        return False, f"Tool '{tool_name}' не найден в модуле (есть: {[a for a in dir(module) if not a.startswith('_')]})"

    def _invoke():
        return tool_func.invoke(test_input)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            result = future.result(timeout=SMOKE_TEST_TIMEOUT)
    except FuturesTimeoutError:
        return False, f"Tool '{tool_name}' завис (таймаут {SMOKE_TEST_TIMEOUT}с)"
    except Exception as e:
        return False, f"Runtime ошибка: {type(e).__name__}: {e}"
    finally:
        sys.modules.pop(module_name, None)

    result_str = str(result)

    if not result_str or result_str.strip() == "":
        return False, "Tool вернул пустой результат"

    error_patterns = [
        "YOUR_API_KEY", "REPLACE_ME", "INSERT_KEY",
        "Unauthorized", "401", "403", "Forbidden",
        "No module named",
    ]
    for pattern in error_patterns:
        if pattern.lower() in result_str.lower():
            return False, f"Tool вернул ошибку: {result_str[:500]}"

    return True, result_str[:500]


def _format_chat_history(state: GeneralGraphState) -> str:
    """Форматирует chat_history в читаемый текст для промптов."""
    history = state.get("chat_history", [])
    if not history:
        return "Нет предыдущей истории."

    previous = history[:-1] if history and history[-1].get("role") == "user" else history

    if not previous:
        return "Нет предыдущей истории."

    lines = []
    for h in previous[-10:]:
        role = "Пользователь" if h["role"] == "user" else "Ассистент"
        content = h["content"][:300]
        lines.append(f"{role}: {content}")

    return "\n".join(lines)