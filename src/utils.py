from .tools.skill_creation import SKILLS_DIR
from .schemas import GeneralGraphState
import importlib
import json
import re
import subprocess
import sys
import importlib.util
from pathlib import Path

SMOKE_TEST_TIMEOUT: int = 15
SMOKE_IMPORT_GRACE: int = 15  # запас wall-времени на импорт langchain в подпроцессе

# Импорт-имя → pip-имя для частых расхождений (авто-установка зависимостей навыков).
MODULE_TO_PKG = {
    "pptx": "python-pptx",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "docx": "python-docx",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "fitz": "pymupdf",
    "Crypto": "pycryptodome",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
}

_MISSING_MODULE_RE = re.compile(r"No module named '([A-Za-z0-9_]+)")


def missing_module_from_error(text: str) -> str:
    """Достаёт имя недостающего модуля из текста ошибки ('' если не про импорт)."""
    m = _MISSING_MODULE_RE.search(text or "")
    return m.group(1) if m else ""


def ensure_python_package(module_name: str) -> tuple[bool, str]:
    """
    Авто-установка python-зависимости навыка через `uv add` (персистентно: pyproject+lock).
    Агент НИКОГДА не должен просить пользователя выполнить pip install сам.
    Возвращает (ok, сообщение).
    """
    import importlib as _il
    import shutil as _shutil

    try:  # уже стоит?
        _il.import_module(module_name)
        return True, f"{module_name} уже установлен"
    except ImportError:
        pass

    uv = _shutil.which("uv")
    if not uv:
        return False, "uv не найден в PATH"
    pkg = MODULE_TO_PKG.get(module_name, module_name)
    try:
        res = subprocess.run([uv, "add", pkg], capture_output=True, text=True, timeout=300)
    except Exception as e:  # noqa: BLE001
        return False, f"uv add {pkg}: {type(e).__name__}: {e}"
    if res.returncode != 0:
        return False, f"uv add {pkg} failed: {(res.stderr or res.stdout)[-300:]}"
    _il.invalidate_caches()
    try:
        _il.import_module(module_name)
        return True, f"установлен {pkg}"
    except ImportError as e:
        return False, f"{pkg} установлен, но '{module_name}' всё равно не импортируется: {e}"


def _skill_loadable(skill_name: str) -> tuple[bool, str]:
    """
    Этап валидации «загружаемость»: модуль навыка обязан импортироваться БЕЗ ошибок
    и содержать хотя бы одну @tool-функцию. Ловит битьё вроде отсутствия
    `from langchain_core.tools import tool` (name 'tool' is not defined) ещё ДО приёма.
    Возвращает (ok, сообщение).
    """
    py_file = SKILLS_DIR / skill_name / f"{skill_name}.py"
    if not py_file.exists():
        return False, "нет файла навыка"
    module_name = f"skills_load_check.{skill_name}"
    try:
        if module_name in sys.modules:
            del sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, str(py_file))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        sys.modules.pop(module_name, None)

    tools = [
        getattr(module, a).name
        for a in dir(module)
        if hasattr(getattr(module, a), "name") and hasattr(getattr(module, a), "invoke")
    ]
    if not tools:
        return False, "не найдено ни одной @tool функции (есть ли 'from langchain_core.tools import tool'?)"
    return True, ", ".join(tools)


# Раннер песочницы: исполняется ОТДЕЛЬНЫМ python-процессом. Ставит resource-лимиты
# (CPU/размер файла/память) ДО загрузки тестируемого кода, грузит модуль, зовёт tool
# и отдаёт результат маркированной JSON-строкой. Падение/убийство процесса не
# затрагивает процесс агента.
_SANDBOX_RUNNER = r"""
import json, sys

def _limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024,) * 2)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3,) * 2)
        except (ValueError, OSError):
            pass  # macOS может не дать RLIMIT_AS — остальные лимиты остаются
    except Exception:
        pass

def _emit(ok, result):
    print("__SMOKE__" + json.dumps({"ok": ok, "result": str(result)[:2000]}, ensure_ascii=False))

_limits()
py_file, tool_name, raw_input = sys.argv[1], sys.argv[2], sys.argv[3]
import importlib.util
try:
    spec = importlib.util.spec_from_file_location("skill_under_test", py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
except Exception as e:
    _emit(False, f"Ошибка загрузки модуля: {type(e).__name__}: {e}"); sys.exit(0)

tool_func = None
for attr_name in dir(module):
    obj = getattr(module, attr_name)
    if hasattr(obj, "name") and hasattr(obj, "invoke") and (obj.name == tool_name or attr_name == tool_name):
        tool_func = obj; break
if tool_func is None:
    _emit(False, f"Tool '{tool_name}' не найден в модуле"); sys.exit(0)

try:
    _emit(True, tool_func.invoke(json.loads(raw_input)))
except Exception as e:
    _emit(False, f"Runtime ошибка: {type(e).__name__}: {e}")
"""


def run_tool_sandboxed(py_file: Path, tool_name: str, test_input: dict,
                       timeout: int = SMOKE_TEST_TIMEOUT) -> tuple[bool, str]:
    """
    Запускает tool из файла в ИЗОЛИРОВАННОМ подпроцессе (отдельный python того же
    venv, resource-лимиты CPU/FSIZE/AS, жёсткий wall-таймаут с kill). Это песочница
    уровня процесса: сгенерированный код не исполняется в процессе агента и не может
    его повесить/уронить. Возвращает (success, result_or_error).
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SANDBOX_RUNNER, str(py_file), tool_name, json.dumps(test_input)],
            capture_output=True, text=True, timeout=timeout + SMOKE_IMPORT_GRACE,
        )
    except subprocess.TimeoutExpired:
        return False, f"Tool '{tool_name}' завис (таймаут {timeout}с) — процесс убит"
    except Exception as e:  # noqa: BLE001
        return False, f"Не удалось запустить песочницу: {type(e).__name__}: {e}"

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__SMOKE__"):
            try:
                payload = json.loads(line[len("__SMOKE__"):])
                return bool(payload["ok"]), str(payload["result"])
            except Exception:  # noqa: BLE001
                break
    err = (proc.stderr or proc.stdout or "").strip()[-500:]
    return False, f"Песочница завершилась без результата (rc={proc.returncode}): {err}"


def _run_smoke_test(skill_name: str, tool_name: str, test_input: dict) -> tuple[bool, str]:
    """
    Smoke-тест навыка В ПЕСОЧНИЦЕ: подпроцесс + resource-лимиты + таймаут.
    Возвращает (success, result_or_error).
    """
    py_file = SKILLS_DIR / skill_name / f"{skill_name}.py"
    if not py_file.exists():
        return False, f"Файл {py_file} не найден"

    success, result_str = run_tool_sandboxed(py_file, tool_name, test_input)
    if not success:
        return False, result_str

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