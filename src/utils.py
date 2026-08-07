from .tools.skill_creation import SKILLS_DIR, _skill_base  # _skill_base: L4 project-навыки
from src.graph.schemas import GeneralGraphState
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path

SMOKE_TEST_TIMEOUT: int = 15
SMOKE_IMPORT_GRACE: int = 15  # запас wall-времени на импорт langchain в подпроцессе

# Маркап тул-вызовов, который модель может эмитить ТЕКСТОМ (deepseek DSML, generic
# <tool_call>): живой прогон показал утечку «<｜DSML｜invoke name="open_url"…>» прямо
# в ответ пользователю. Это не ответ — это несостоявшийся вызов инструмента.
_TOOL_MARKUP_RE = re.compile(
    r"<[^<>]*｜DSML｜[^<>]*>|</?tool_calls?>|<｜tool[▁_]calls?[^>]*｜>|"
    r"</?(?:invoke|parameter)\b[^<>]*>",
    re.IGNORECASE)


def strip_tool_markup(text: str) -> str:
    """Убирает маркап тул-вызовов из текста ответа. Если после чистки остались лишь
    обрывки аргументов (URL/имена параметров) — текст не был ответом, вернём ''."""
    if not text:
        return ""
    low = text.lower()
    if "｜dsml｜" not in low and "<tool_call" not in low and "<invoke" not in low:
        return text
    cleaned = _TOOL_MARKUP_RE.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # остался служебный мусор вызова, а не человеческий ответ → пусто (вызывающий сделает итог)
    if len(cleaned) < 12 or re.fullmatch(r"[\w/:.?=+&%~#-]+", cleaned):
        return ""
    return cleaned

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
        # Таймаут 120с (не 300): `uv add` резолвит весь lock и может надолго подвиснуть на
        # «Resolving dependencies…», морозя интерактивную задачу. Лучше быстро сдаться с
        # честным сообщением, чем 5-минутный фриз (живой баг: заказ суши завис на установке).
        res = subprocess.run([uv, "add", pkg], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, (f"uv add {pkg}: установка зависимости заняла >120с и прервана — навык "
                       "не стоит этой задержки; реши задачу без него (например, через браузер/веб-поиск)")
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


# Раннер проверки загружаемости: импорт модуля навыка + перечисление @tool-функций,
# исполняется в ОТДЕЛЬНОМ процессе (rlimits + опц. syscall-sandbox). Раньше exec_module
# шёл в процессе агента — module-level код LLM-навыка исполнялся в хосте ещё до HITL
# (Hermes-порт: изоляция исполнения). Формат вывода тот же __SMOKE__-JSON.
_LOADCHECK_RUNNER = r"""
import json, sys

def _limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024,) * 2)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3,) * 2)
        except (ValueError, OSError):
            pass
    except Exception:
        pass

def _emit(ok, result):
    print("__SMOKE__" + json.dumps({"ok": ok, "result": str(result)[:2000]}, ensure_ascii=False))

_limits()
py_file = sys.argv[1]
import importlib.util
try:
    spec = importlib.util.spec_from_file_location("skill_load_check", py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
except Exception as e:
    _emit(False, f"{type(e).__name__}: {e}"); sys.exit(0)

tools = [getattr(module, a).name for a in dir(module)
         if hasattr(getattr(module, a), "name") and hasattr(getattr(module, a), "invoke")]
if not tools:
    _emit(False, "не найдено ни одной @tool функции (есть ли 'from langchain_core.tools import tool'?)")
else:
    _emit(True, ", ".join(tools))
"""


def _skill_loadable(skill_name: str) -> tuple[bool, str]:
    """
    Этап валидации «загружаемость»: модуль навыка обязан импортироваться БЕЗ ошибок
    и содержать хотя бы одну @tool-функцию. Ловит битьё вроде отсутствия
    `from langchain_core.tools import tool` (name 'tool' is not defined) ещё ДО приёма.
    Импорт идёт в ИЗОЛИРОВАННОМ подпроцессе (rlimits + опц. syscall-sandbox) — module-level
    код свежесгенерированного навыка НЕ исполняется в процессе агента.
    Возвращает (ok, сообщение).
    """
    py_file = _skill_base(skill_name) / f"{skill_name}.py"
    if not py_file.exists():
        return False, "нет файла навыка"

    base = [_python_exe(), "-c", _LOADCHECK_RUNNER, str(py_file)]
    cmd = _syscall_sandbox_prefix() + base
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=SMOKE_TEST_TIMEOUT + SMOKE_IMPORT_GRACE)
    except FileNotFoundError:
        proc = subprocess.run(base, capture_output=True, text=True,
                              timeout=SMOKE_TEST_TIMEOUT + SMOKE_IMPORT_GRACE)
    except subprocess.TimeoutExpired:
        return False, f"импорт навыка завис (таймаут {SMOKE_TEST_TIMEOUT + SMOKE_IMPORT_GRACE}с) — процесс убит"
    except Exception as e:  # noqa: BLE001
        return False, f"песочница проверки не запустилась: {type(e).__name__}: {e}"

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__SMOKE__"):
            try:
                payload = json.loads(line[len("__SMOKE__"):])
                return bool(payload["ok"]), str(payload["result"])
            except Exception:  # noqa: BLE001
                break
    err = (proc.stderr or proc.stdout or "").strip()[-400:]
    return False, f"проверка загружаемости без результата (rc={proc.returncode}): {err}"


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


def _syscall_sandbox_prefix(no_net: bool = False) -> list[str]:
    """
    Опциональная изоляция syscall-уровня ПОВЕРХ rlimits, если в системе есть
    подходящий инструмент (best-effort, без жёсткой зависимости):
      • Linux: bubblewrap (bwrap) — read-only / namespaces, либо firejail;
      • macOS: sandbox-exec с профилем (deprecated, но рабочий) — ВКЛЮЧЁН по умолчанию
        (запись только в $TMPDIR + опц. deny network); выключить — AGENT_SANDBOX_EXEC=0
        (если ломает редкий импорт, пишущий кэш вне tmp).
    Управление: AGENT_SYSCALL_SANDBOX=0 — выключить совсем; =1/auto (default) — авто.
    Возвращает prefix-команду (или пусто → только rlimits).
    """
    mode = os.environ.get("AGENT_SYSCALL_SANDBOX", "auto").lower()
    if mode in ("0", "off", "none"):
        return []
    import platform as _pf

    # no_net=True → ОТРЕЗАЕМ сеть (анти-эксфильтрация недоверенным навыком: AST разрешает urllib/
    # socket, и без сети их канал утечки закрыт). Default — сеть ВКЛ (генерируемые навыки часто
    # ходят в API; иначе режем способность). Флаг AGENT_SKILL_SANDBOX_NO_NET=1 — полный lockdown.
    if _pf.system() == "Linux":
        if shutil.which("bwrap"):
            # read-only весь корень, изоляция pid/ipc/uts, без новых привилегий; /tmp как tmpfs.
            cmd = [
                "bwrap", "--ro-bind", "/", "/", "--tmpfs", "/tmp",
                "--unshare-pid", "--unshare-ipc", "--unshare-uts",
                "--die-with-parent", "--new-session",
            ]
            if no_net:
                cmd.insert(1, "--unshare-net")   # без сетевого неймспейса → нет egress
            return cmd
        if shutil.which("firejail"):
            return ["firejail", "--quiet", "--private-tmp", "--noprofile"] + (["--net=none"] if no_net else [])
    elif _pf.system() == "Darwin" and os.environ.get("AGENT_SANDBOX_EXEC", "1") != "0":
        if shutil.which("sandbox-exec"):
            # ВКЛ по умолчанию (opt-out AGENT_SANDBOX_EXEC=0). Профиль: чтение ок, запрет записи вне
            # tmp (+$TMPDIR — туда пишут кэши некоторых либ на импорте), опц. запрет сети. Так
            # ФАКТИЧЕСКОЕ поведение python_exec совпадает с КОНТРАКТОМ «без сети/ФС».
            tmp = os.environ.get("TMPDIR", "").rstrip("/")
            tmp_rule = f'(subpath "{tmp}")' if tmp else ""
            prof = ("(version 1)(allow default)(deny file-write*)"
                    f'(allow file-write* (subpath "/private/tmp") (subpath "/tmp") {tmp_rule})'
                    + ("(deny network*)" if no_net else ""))
            return ["sandbox-exec", "-p", prof]
    return []


def _python_exe() -> str:
    """Интерпретатор для подпроцесса-песочницы. ВАЖНО для упакованного приложения (PyInstaller):
    там sys.executable = САМ бинарь приложения, и `[sys.executable, "-c", код]` ПЕРЕЗАПУСКАЕТ
    приложение вместо исполнения python-кода (баг «приложение перезапустилось» на запросе). В
    frozen берём реальный системный python3 (для python_exec достаточно — он stdlib-only). Из
    исходников — обычный venv-python."""
    import shutil
    import sys as _sys
    if not getattr(_sys, "frozen", False):
        return _sys.executable
    for cand in (shutil.which("python3"), shutil.which("python"),
                 "/usr/bin/python3", "/usr/local/bin/python3", "/opt/homebrew/bin/python3"):
        if cand and os.path.exists(cand):
            return cand
    return _sys.executable  # крайний случай (лучше, чем ничего)


def run_tool_sandboxed(py_file: Path, tool_name: str, test_input: dict,
                       timeout: int = SMOKE_TEST_TIMEOUT, no_net: bool = False) -> tuple[bool, str]:
    """
    Запускает tool из файла в ИЗОЛИРОВАННОМ подпроцессе (отдельный python того же
    venv, resource-лимиты CPU/FSIZE/AS, жёсткий wall-таймаут с kill) + опциональная
    syscall-изоляция (bubblewrap/firejail/sandbox-exec, если есть). Сгенерированный
    код не исполняется в процессе агента. no_net=True → отрезаем сеть (анти-эксфильтрация).
    Возвращает (success, result_or_error).
    """
    base = [_python_exe(), "-c", _SANDBOX_RUNNER, str(py_file), tool_name, json.dumps(test_input)]
    cmd = _syscall_sandbox_prefix(no_net) + base
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + SMOKE_IMPORT_GRACE)
    except FileNotFoundError:
        # sandbox-обёртка вдруг недоступна → деградируем до чистых rlimits
        proc = subprocess.run(base, capture_output=True, text=True, timeout=timeout + SMOKE_IMPORT_GRACE)
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


# ── Вычислительный слой: исполнение произвольного Python в песочнице ──────────────
# Агент с исполнением кода кратно способнее: считать/агрегировать/парсить данные,
# которые нашёл research (статистика, числа, фильтры). Код идёт в ИЗОЛИРОВАННЫЙ
# подпроцесс (rlimits CPU/mem/FSIZE + опц. syscall-изоляция + wall-kill) — НЕ в процессе
# агента. Это закрывает «вычислительный» пробел held-out (rain-probability, числовые GAIA).
_PYEXEC_RUNNER = r"""
import json, sys, io, contextlib
def _limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024,) * 2)
        try: resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3,) * 2)
        except (ValueError, OSError): pass
    except Exception: pass
_limits()
code = open(sys.argv[1], encoding="utf-8").read()
buf = io.StringIO(); ok = True; err = ""
try:
    with contextlib.redirect_stdout(buf):
        exec(code, {"__name__": "__main__"})
except Exception as e:
    ok = False; err = f"{type(e).__name__}: {e}"
print("__PYEXEC__" + json.dumps({"ok": ok, "stdout": buf.getvalue()[:4000], "error": err}, ensure_ascii=False))
"""


def run_python_sandboxed(code: str, timeout: int = 12, no_net: bool = False) -> tuple[bool, str]:
    """Исполняет произвольный Python в ИЗОЛИРОВАННОМ подпроцессе. (ok, stdout|ошибка).
    no_net=True → отрезаем сеть (python_exec по контракту «без сети/ФС» + он самый широкий канал
    эксфильтрации: всегда доступен, без HITL, код приходит от LLM/инъекции). Эффективно при syscall-
    песочнице (bwrap/firejail/sandbox-exec); на голом macOS — no-op (только rlimits)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(code)
        path = fh.name
    base = [_python_exe(), "-c", _PYEXEC_RUNNER, path]
    cmd = _syscall_sandbox_prefix(no_net) + base
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        proc = subprocess.run(base, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"Код завис (таймаут {timeout}с) — процесс убит"
    except Exception as e:  # noqa: BLE001
        return False, f"Песочница не запустилась: {type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(path)
        except Exception:  # noqa: BLE001
            pass
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__PYEXEC__"):
            p = json.loads(line[len("__PYEXEC__"):])
            if p["ok"]:
                return True, (p["stdout"].strip() or "(код отработал, но ничего не вывел — используй print())")
            return False, p["error"]
    return False, (proc.stderr or "").strip()[-400:] or "нет результата от песочницы"


def _run_smoke_test(skill_name: str, tool_name: str, test_input: dict) -> tuple[bool, str]:
    """
    Smoke-тест навыка В ПЕСОЧНИЦЕ: подпроцесс + resource-лимиты + таймаут.
    Возвращает (success, result_or_error).
    """
    py_file = _skill_base(skill_name) / f"{skill_name}.py"
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