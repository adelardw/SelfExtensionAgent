import ast
import json
import os
import importlib.util
import sys
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool

# Атомарная+сериализованная запись реестра навыков: фоновый reflect-поток судит/удаляет навыки
# (_delete_skill_impl → _save_reg_at) ПАРАЛЛЕЛЬНО с основным (create/sync) → голый write_text давал
# read-modify-write гонку на одном JSON (тот же класс, что 2c в intent/prompt_store). Lock + temp→
# fsync→os.replace: читатель видит старый ИЛИ новый ЦЕЛЫЙ файл.
_REG_LOCK = threading.Lock()


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    with _REG_LOCK:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def _skills_base() -> Path:
    """
    BASE_PATH навыков/тулов: env AGENT_BASE_PATH → config.yml skills.base_path → src/skills.
    Позволяет держать навыки/подключаемые тулы в произвольном каталоге.
    """
    base = os.getenv("AGENT_BASE_PATH")
    if not base:
        try:
            from omegaconf import OmegaConf

            base = OmegaConf.load("config.yml").get("skills", {}).get("base_path")
        except Exception:  # noqa: BLE001
            base = None
    return Path(base) if base else Path("src/skills")


SKILLS_DIR = _skills_base()
REGISTRY_FILE = SKILLS_DIR / "registry.json"


def _load_protected() -> set[str]:
    """Базовые защищённые навыки из config.yml (skills.protected)."""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("config.yml")
        return set(cfg.get("skills", {}).get("protected", []) or [])
    except Exception:  # noqa: BLE001
        return set()


PROTECTED_SKILLS: set[str] = _load_protected()


def _is_protected(name: str, registry: Optional[dict] = None) -> bool:
    """Навык защищён, если он в config-списке ИЛИ помечен protected в реестре."""
    if name in PROTECTED_SKILLS:
        return True
    reg = registry if registry is not None else _load_registry()
    return bool(reg.get(name, {}).get("protected"))


def _ensure_dirs():
    """Создаёт корневую директорию скиллов и registry если нет."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        REGISTRY_FILE.write_text(json.dumps({}, indent=2, ensure_ascii=False))


def _load_registry() -> dict:
    _ensure_dirs()
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def _save_registry(registry: dict):
    _ensure_dirs()
    _atomic_write_json(REGISTRY_FILE, registry)   # лок + atomic (гонка фон-reflect ↔ main)


# ── L4: проектный ярус навыков (.sea/skills/) ──────────────────────────────────
# Навыки трёхъярусны (как память): ГЛОБАЛЬНЫЕ/user (src/skills + registry.json, кросс-проект),
# ПРОЕКТНЫЕ (агент создал для проекта → .sea/skills/, project-local), ВНЕШНИЕ (SKILL.md).
# Разделение read/write: загрузчики читают ОБЪЕДИНЁННЫЙ реестр; _load/_save_registry — ТОЛЬКО
# глобальный (проектные навыки НЕ протекают в global registry). Нет .sea/skills → всё как было.
def _project_skills_dir() -> Optional[Path]:
    base = Path(os.getenv("AGENT_PROJECT_ROOT") or Path.cwd()) / ".sea" / "skills"
    return base if base.is_dir() else None


def _skill_base(name: str) -> Path:
    """Каталог навыка: ПРОЕКТНЫЙ (.sea/skills) приоритетнее ГЛОБАЛЬНОГО (src/skills)."""
    pj = _project_skills_dir()
    if pj and (pj / name).is_dir():
        return pj / name
    return SKILLS_DIR / name


def _merged_registry() -> dict:
    """Реестр для ЧТЕНИЯ (загрузчики/селектор): глобальный + проектный поверх."""
    reg = dict(_load_registry())
    pj = _project_skills_dir()
    if pj and (pj / "registry.json").exists():
        try:
            reg.update(json.loads((pj / "registry.json").read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return reg


# ── L4b/c: создание навыков ПО СКОУПУ (project/global) + роутинг реестра ────────
def _sea_initialized() -> bool:
    """Проект инициализирован (`sea init` создал .sea/) → дефолт создания навыков = project."""
    return (Path(os.getenv("AGENT_PROJECT_ROOT") or Path.cwd()) / ".sea").is_dir()


def _project_skills_root() -> Path:
    """Целевой каталог проектных навыков (может ещё не существовать — для СОЗДАНИЯ)."""
    return Path(os.getenv("AGENT_PROJECT_ROOT") or Path.cwd()) / ".sea" / "skills"


def _project_registry_path() -> Path:
    return _project_skills_root() / "registry.json"


def _load_reg_at(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _save_reg_at(path: Path, reg: dict) -> None:
    _atomic_write_json(path, reg)                 # лок + atomic (тот же _REG_LOCK)


def _skill_scope(name: str) -> str:
    """'project' если навык лежит в .sea/skills, иначе 'global'."""
    pj = _project_skills_dir()
    return "project" if (pj and (pj / name).is_dir()) else "global"


def _registry_path_for(name: str) -> Path:
    """Реестр, которому ПРИНАДЛЕЖИТ существующий навык (read/delete/mark/update пишут сюда)."""
    return _project_registry_path() if _skill_scope(name) == "project" else REGISTRY_FILE


def _default_scope() -> str:
    """Дефолт скоупа создаваемого навыка: project в инициализированном проекте, иначе global."""
    return "project" if _sea_initialized() else "global"


def _validate_python(code: str) -> tuple[bool, str]:
    """Проверяет синтаксис Python-кода БЕЗ выполнения."""
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _security_gate(code: str) -> tuple[bool, str]:
    """
    AST-гейт безопасности для ГЕНЕРИРУЕМОГО кода (точка записи — единственный путь,
    которым LLM сохраняет код навыка). Владелец может отключить: AGENT_ALLOW_RISKY_SKILLS=1.
    """
    if os.getenv("AGENT_ALLOW_RISKY_SKILLS") == "1":
        return True, "OK (гейт отключён env)"
    from src.tools.utils_validation import validate_skill_code

    ok, issues = validate_skill_code(code)
    if ok:
        return True, "OK"
    return False, (
        "Security gate: " + "; ".join(issues) +
        ". Генерируемым навыкам запрещены subprocess/os.system/eval/exec и т.п. — "
        "используй стандартные библиотеки (urllib, json, pathlib) или существующие core-навыки."
    )


# Имена навыков, созданных в этой сессии (структурный канал для create_skills_node,
# вместо хрупкого регэкспа по тексту сообщений агента).
_session_created: list[str] = []


def pop_last_created() -> str:
    """Возвращает имя последнего созданного в сессии навыка (и забывает его)."""
    return _session_created.pop() if _session_created else ""


# ── временные навыки (создан под задачу → решение «оставить/выбросить» позже) ──

def mark_temporary(name: str) -> None:
    """Помечает навык временным: создан под текущую задачу, в библиотеку ещё не принят."""
    path = _registry_path_for(name)  # L4: реестр по скоупу навыка (project/global)
    registry = _load_reg_at(path)
    if name in registry:
        registry[name]["temporary"] = True
        _save_reg_at(path, registry)


def clear_temporary(name: str) -> None:
    """Принимает навык в библиотеку насовсем (решение retention-судьи)."""
    path = _registry_path_for(name)
    registry = _load_reg_at(path)
    if name in registry and registry[name].pop("temporary", None):
        _save_reg_at(path, registry)



@tool("list_skills")
def list_skills() -> str:
    """
    List all available skills with their descriptions and status.
    Use this FIRST to check what skills already exist before creating new ones.

    Returns:
        str: A formatted list of all registered skills.
    """
    registry = _merged_registry()  # L4: глобальные + проектные
    if not registry:
        return "No skills registered yet."

    lines = []
    for name, meta in registry.items():
        status = "ready" if meta.get("has_tools") else "description only"
        lock = " 🔒core" if _is_protected(name, registry) else ""
        temp = " 🕒temp" if meta.get("temporary") else ""
        scope = " 📁project" if _skill_scope(name) == "project" else ""
        imp = " 🦞imported" if meta.get("imported") else ""
        lines.append(
            f"• {name} [{status}]{lock}{scope}{temp}{imp} — {meta['description'][:100]}"
        )
    return "Available skills:\n" + "\n".join(lines)


@tool("read_skill")
def read_skill(name: str) -> str:
    """
    Read the full content of a skill (description + tool code).
    Use this to understand what an existing skill does before using or modifying it.

    Args:
        name: The exact name of the skill to read.

    Returns:
        str: The skill's description and tool source code.
    """
    skill_dir = _skill_base(name)  # L4: project-навык приоритетнее global
    parts = [f"=== Skill: {name} ==="]

    md_file = skill_dir / f"{name}.md"
    if md_file.exists():
        parts.append(f"\n## Description:\n{md_file.read_text(encoding='utf-8')}")

    prompt_file = skill_dir / "prompt.md"
    if prompt_file.exists():
        parts.append(f"\n## System Prompt:\n{prompt_file.read_text(encoding='utf-8')}")

    py_file = skill_dir / f"{name}.py"
    if py_file.exists():
        parts.append(f"\n## Tools code:\n```python\n{py_file.read_text(encoding='utf-8')}\n```")

    if len(parts) == 1:
        return f"Skill '{name}' not found."

    return "\n".join(parts)


@tool("create_skill")
def create_skill(
    name: str,
    description: str,
    tool_code: Optional[str] = None,
    system_prompt: Optional[str] = None,
    scope: Optional[str] = None,
) -> str:
    """
    Create a new skill with description, system prompt, and optionally tool code.

    Args:
        name: Short snake_case name for the skill (e.g. 'web_search', 'data_analysis').
        description: Markdown description of WHEN and HOW to use this skill.
            Include: purpose, triggers, inputs/outputs, examples.
        tool_code: Python source code with @tool-decorated functions.
            Must be valid Python. Will be validated before saving.
            Can be omitted and added later via update_skill_tools.
        system_prompt: System prompt that will be INJECTED into the execution agent
            when this skill is active.
        scope: 'project' — навык живёт в .sea/skills/ (только этот проект); 'global' —
            в библиотеке пользователя (кросс-проект, src/skills). По умолчанию: project в
            инициализированном проекте (есть .sea/), иначе global. ПРОЕКТНЫЙ — для штук,
            нужных ЭТОМУ проекту; GLOBAL — для общеполезных способностей.

    Returns:
        str: Confirmation or error message.
    """
    _ensure_dirs()
    # L4b: скоуп → каталог + реестр. Дефолт project в проекте, иначе global (аддитивно).
    scope = scope if scope in ("project", "global") else _default_scope()
    reg_path = _project_registry_path() if scope == "project" else REGISTRY_FILE
    base_dir = _project_skills_root() if scope == "project" else SKILLS_DIR
    registry = _load_reg_at(reg_path)

    if name in registry:
        return (
            f"Skill '{name}' already exists (scope: {scope}). "
            f"Use 'read_skill' to inspect it, or 'update_skill_tools' to modify."
        )

    if tool_code:
        is_valid, err = _validate_python(tool_code)
        if not is_valid:
            return f"Invalid Python code — {err}. Fix the code and try again."
        safe, sec_msg = _security_gate(tool_code)
        if not safe:
            return f"Rejected. {sec_msg}"

    skill_dir = base_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    (skill_dir / f"{name}.md").write_text(description, encoding="utf-8")

    has_tools = False
    if tool_code:
        (skill_dir / f"{name}.py").write_text(tool_code, encoding="utf-8")
        has_tools = True

    if system_prompt:
        (skill_dir / "prompt.md").write_text(system_prompt, encoding="utf-8")

    registry[name] = {
        "description": description[:200],
        "has_tools": has_tools,
        "has_system_prompt": bool(system_prompt),
        "scope": scope,                       # L4: ярус навыка
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "version": 1,
    }
    _save_reg_at(reg_path, registry)
    _session_created.append(name)

    result = f"Skill '{name}' created successfully (scope: {scope})."
    if has_tools:
        result += " Tools are ready to be loaded."
    else:
        result += " Add tools later with 'update_skill_tools'."
    if system_prompt:
        result += " System prompt saved."
    return result


@tool("update_skill_tools")
def update_skill_tools(name: str, tool_code: str, append: bool = False) -> str:
    """
    Update or add tool code for an existing skill.

    Args:
        name: The name of the skill to update.
        tool_code: New Python source code with @tool-decorated functions.
        append: If True, append code to existing file. If False, overwrite.

    Returns:
        str: Confirmation or error message.
    """
    _rpath = _registry_path_for(name)  # L4: реестр по скоупу навыка
    registry = _load_reg_at(_rpath)
    if name not in registry:
        return f"Skill '{name}' does not exist. Create it first with 'create_skill'."

    if _is_protected(name, registry) and not append:
        return (
            f"Skill '{name}' is PROTECTED (core capability). Overwriting its tools is "
            f"blocked. Use append=True to add tools, or pick a different skill name."
        )

    is_valid, err = _validate_python(tool_code)
    if not is_valid:
        return f"Invalid Python code — {err}. Fix and retry."
    safe, sec_msg = _security_gate(tool_code)
    if not safe:
        return f"Rejected. {sec_msg}"

    skill_file = _skill_base(name) / f"{name}.py"

    if append and skill_file.exists():
        existing = skill_file.read_text(encoding="utf-8")
        combined = existing + "\n\n" + tool_code
        is_valid, err = _validate_python(combined)
        if not is_valid:
            return f"Appended code creates conflicts — {err}."
        skill_file.write_text(combined, encoding="utf-8")
    else:
        skill_file.write_text(tool_code, encoding="utf-8")

    registry[name]["has_tools"] = True
    registry[name]["updated_at"] = datetime.now().isoformat()
    registry[name]["version"] = registry[name].get("version", 0) + 1
    _save_reg_at(_rpath, registry)

    return f"Tools for skill '{name}' updated (v{registry[name]['version']})."


def _delete_skill_impl(name: str, allow_protected: bool) -> str:
    import shutil

    _rpath = _registry_path_for(name)  # L4: реестр по скоупу навыка
    registry = _load_reg_at(_rpath)
    skill_dir = _skill_base(name)

    if name not in registry and not skill_dir.exists():
        return f"Skill '{name}' not found."

    if _is_protected(name, registry) and not allow_protected:
        return (
            f"Skill '{name}' is PROTECTED (core capability) and cannot be deleted by the agent. "
            f"Only the owner can remove it (force_delete_skill from CLI)."
        )

    if skill_dir.exists():
        shutil.rmtree(skill_dir)

    registry.pop(name, None)
    _save_reg_at(_rpath, registry)
    _MODULE_CACHE.pop(name, None)  # не держим в кэше удалённый навык

    return f"Skill '{name}' has been deleted."


@tool("delete_skill")
def delete_skill(name: str) -> str:
    """
    Delete a non-protected skill entirely (description + tools + registry entry).
    Use with caution — this is irreversible. Protected (core) skills can NEVER
    be deleted through this tool.

    Args:
        name: The name of the skill to delete.

    Returns:
        str: Confirmation message.
    """
    return _delete_skill_impl(name, allow_protected=False)


def force_delete_skill(name: str) -> str:
    """Владельческое удаление (включая protected) — НЕ tool, только из кода/CLI."""
    return _delete_skill_impl(name, allow_protected=True)


# Кэш загруженных модулей навыков: name -> (mtime, module). Без него
# get_all_loaded_skill_tools ре-exec'ил бы каждый навык на КАЖДОМ шаге графа —
# впустую и опасно при import-side-effects. Инвалидация по mtime файла.
_MODULE_CACHE: dict[str, tuple[float, object]] = {}


def _trusted_skill(name: str) -> bool:
    """Core/protected навыки писал автор продукта — им доверяем (могут законно использовать
    subprocess/osascript). Сгенерированные/импортированные/orphan — НЕ доверяем: их module-level
    код гейтим тем же AST-гейтом, что и путь записи, ПЕРЕД exec_module."""
    if name in PROTECTED_SKILLS:
        return True
    return bool(_merged_registry().get(name, {}).get("protected"))


def _load_skill_module(name: str, py_file: Path) -> tuple[object | None, str]:
    """Безопасно загрузить модуль навыка для извлечения @tool-функций.

    Закрывает дыру «exec до HITL»: HITL гейтит ВЫЗОВ тула, но module-level код untrusted-навыка
    исполняется уже при импорте. Поэтому для недоверенных навыков ПЕРЕД exec_module прогоняем тот
    же AST-гейт, что и на пути записи (паритет write/exec; orphan и imported код больше не
    исполняется без проверки). Плюс кэш по mtime — не ре-exec'им на каждом шаге.

    Возвращает (module | None, reason). AST-гейт обходим перефразировкой (как и на write-пути);
    это паритет, а не песочница.
    """
    try:
        mtime = py_file.stat().st_mtime
    except OSError as e:  # noqa: BLE001
        return None, f"stat failed: {e}"

    cached = _MODULE_CACHE.get(name)
    if cached and cached[0] == mtime:
        return cached[1], "OK (cached)"

    code = py_file.read_text(encoding="utf-8")
    is_valid, err = _validate_python(code)
    if not is_valid:
        return None, f"invalid code: {err}"

    # Гейт ПЕРЕД exec_module (закрывает «exec до HITL»), три уровня доверия:
    #   • core/protected — автор продукта, доверяем (subprocess для osascript и т.п.);
    #   • imported (OpenClaw-обёртка CLI) — своя модель (HITL + allowlist), но module-level
    #     код всё равно исполнится при импорте → проверяем только уровень модуля;
    #   • прочее (сгенерированное/orphan на диске) — полный AST-гейт (контракт «чистый stdlib»).
    if not _trusted_skill(name):
        meta = _merged_registry().get(name, {})
        if meta.get("imported"):
            from src.tools.utils_validation import validate_module_level
            ok, issues = validate_module_level(code)
            if not ok:
                return None, "Module-level gate: " + "; ".join(issues)
        else:
            safe, sec_msg = _security_gate(code)
            if not safe:
                return None, sec_msg

    module_name = f"skills.{name}"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001
        sys.modules.pop(module_name, None)
        return None, f"{type(e).__name__}: {e}"
    _MODULE_CACHE[name] = (mtime, module)
    return module, "OK"


RUNTIME_SANDBOX_TIMEOUT = int(os.getenv("AGENT_SKILL_SANDBOX_TIMEOUT") or 30)


def _should_sandbox(name: str, meta: dict) -> bool:
    """Исполнять ли ВЫЗОВ навыка в рантайме через subprocess-песочницу (а не in-process).
    Да — для сгенерированного/orphan кода: AST-гейт обходим (urllib/open разрешены), in-process
    вызов имел бы права процесса агента (читать ~/.ssh/.env). Нет — для core/protected (доверен,
    его писал автор; device_control нужен полный доступ) и imported (своя allowlist+HITL модель)."""
    if _trusted_skill(name):
        return False
    if meta.get("imported"):
        return False
    return True


def _sandbox_wrap(tool_obj, skill_name: str, py_file: Path):
    """Обернуть @tool недоверенного навыка: его ВЫЗОВ идёт в ОТДЕЛЬНОМ подпроцессе
    (run_tool_sandboxed), а не в процессе агента. Тело тула не исполняется in-process — только
    метаданные (name/description/schema) берём из модуля (его module-level код прошёл AST-гейт).

    ЧЕСТНЫЕ ГРАНИЦЫ по платформе (важно):
      • ВСЕГДА: rlimits CPU/mem/FSIZE + wall-kill — против runaway/исчерпания ресурсов;
      • запрет записи вне /tmp, изоляция ФС-ЧТЕНИЯ (~/.ssh/.env) и сети — ТОЛЬКО при syscall-
        песочнице (Linux bwrap/firejail; macOS sandbox-exec под AGENT_SANDBOX_EXEC=1). Без них
        (напр. голый macOS) подпроцесс МОЖЕТ читать файлы и ходить в сеть → эксфильтрация возможна.
        Полный lockdown сети: AGENT_SKILL_SANDBOX_NO_NET=1.
    То есть это изоляция ПРОЦЕССА (не in-process), но НЕ полный ФС-сэндбокс на платформе без bwrap/
    sandbox-exec — сознательный потолок «своя машина владельца» (долг ревью #2)."""
    from langchain_core.tools import StructuredTool
    from ..utils import run_tool_sandboxed   # lazy: utils импортирует skill_creation (анти-цикл)

    tname = getattr(tool_obj, "name", skill_name)
    # AGENT_SKILL_SANDBOX_NO_NET=1 → полный lockdown (без сети): закрывает urllib/socket-эксфильтрацию
    # недоверенным навыком. Default off — генерируемые навыки часто ходят в API (иначе режем способность).
    no_net = os.getenv("AGENT_SKILL_SANDBOX_NO_NET") == "1"

    async def _arun(**kwargs):
        import asyncio
        ok, result = await asyncio.to_thread(
            run_tool_sandboxed, py_file, tname, kwargs, RUNTIME_SANDBOX_TIMEOUT, no_net)
        return result if ok else f"[sandbox] навык '{tname}' не выполнен: {result}"

    return StructuredTool(
        name=tname,
        description=getattr(tool_obj, "description", "") or tname,
        args_schema=getattr(tool_obj, "args_schema", None),
        coroutine=_arun,
    )


@tool("load_skill_tools")
def load_skill_tools(name: str) -> str:
    """
    Dynamically load tools from a skill's .py file so they can be used immediately.
    Call this AFTER creating a skill to make its tools available in the current session.

    Args:
        name: The name of the skill whose tools to load.

    Returns:
        str: List of loaded tool names or error message.
    """
    py_file = _skill_base(name) / f"{name}.py"  # L4: project-навык приоритетнее
    if not py_file.exists():
        return f"Skill '{name}' has no tools file. Create tools first."

    module, reason = _load_skill_module(name, py_file)
    if module is None:
        return f"Cannot load '{name}': {reason}"

    loaded_tools = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if hasattr(obj, "name") and hasattr(obj, "invoke"):
            loaded_tools.append(obj.name)

    if loaded_tools:
        return (
            f"Skill '{name}' loaded. Available tools: {', '.join(loaded_tools)}. "
            f"You can now use these tools."
        )
    return (
        f"Skill '{name}' loaded but no @tool functions found. "
        f"Make sure functions are decorated with @tool."
    )


@tool("get_skills_for_prompt")
def get_skills_for_prompt() -> str:
    """
    Get all skill descriptions formatted for injection into the system prompt.
    Use this to understand what capabilities are currently available.

    Returns:
        str: Combined markdown descriptions of all registered skills.
    """
    registry = _merged_registry()  # L4: глобальные + проектные
    if not registry:
        return "No skills available."

    sections = []
    for name, meta in registry.items():
        md_file = _skill_base(name) / f"{name}.md"
        if md_file.exists():
            content = md_file.read_text(encoding="utf-8")
            status = "tools ready" if meta.get("has_tools") else "no tools yet"
            sections.append(f"### Skill: {name} ({status})\n{content}")

    return "\n\n---\n\n".join(sections)


# Порог: пока навыков мало — показываем все (дёшево). При росте библиотеки селектор
# захлёбывается полным списком → включаем ToolSearch (retrieval топ-релевантных).
TOOLSEARCH_THRESHOLD = 12
TOOLSEARCH_TOP = 8


def get_relevant_skills_for_prompt(query: str, top: int = TOOLSEARCH_TOP) -> str:
    """
    ToolSearch: BM25-retrieval НАВЫКОВ по запросу (вместо показа ВСЕХ селектору). Масштабирует
    выбор инструментов при росте библиотеки. Reuse канонического ранкера (src.retrieval).
    Если навыков мало (< порога) или нет совпадений — фолбэк на полный список.
    """
    registry = _merged_registry()  # L4: глобальные + проектные навыки
    if not registry or len(registry) < TOOLSEARCH_THRESHOLD:
        return get_skills_for_prompt.invoke({})  # это @tool, не функция

    from src.search.retrieval import bm25_rank

    names, docs, sections = [], [], []
    for name, meta in registry.items():
        md_file = _skill_base(name) / f"{name}.md"
        if not md_file.exists():
            continue
        content = md_file.read_text(encoding="utf-8")
        status = "tools ready" if meta.get("has_tools") else "no tools yet"
        names.append(name)
        docs.append(f"{name} {content}")
        sections.append(f"### Skill: {name} ({status})\n{content}")

    idx = bm25_rank(docs, query, top)
    if not idx:
        return get_skills_for_prompt.invoke({})  # это @tool, не функция
    return "\n\n---\n\n".join(sections[i] for i in idx)




def get_skill_runtime_prompts(names: list[str]) -> str:
    """
    Возвращает объединённые системные промпты для указанных навыков.
    Используется при инъекции в execution-агента.
    Если у навыка нет prompt.md, фолбэчит на description (.md).
    """
    parts = []
    for name in names:
        prompt_file = _skill_base(name) / "prompt.md"  # L4: проектный приоритетнее
        if prompt_file.exists():
            parts.append(
                f"[Навык: {name}]\n{prompt_file.read_text(encoding='utf-8')}"
            )
        else:
            md_file = _skill_base(name) / f"{name}.md"
            if md_file.exists():
                parts.append(
                    f"[Навык: {name} (описание)]\n{md_file.read_text(encoding='utf-8')}"
                )
    return "\n\n---\n\n".join(parts) if parts else ""


def sync_registry() -> dict:
    """
    Автообновление реестра навыков (вызывать при старте).
      • orphan на диске (есть папка+код, но нет в registry) → регистрируется;
      • битая запись (есть в registry, но папки нет) → удаляется;
      • защищённые из config помечаются protected=True.
    Возвращает сводку изменений.
    """
    _ensure_dirs()
    registry = _load_registry()
    added, removed, protected = [], [], []

    # 1. orphan-скиллы на диске → в реестр
    for sub in SKILLS_DIR.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        has_py = (sub / f"{name}.py").exists()
        md = sub / f"{name}.md"
        if name not in registry and (has_py or md.exists()):
            desc = md.read_text(encoding="utf-8")[:200] if md.exists() else ""
            registry[name] = {
                "description": desc,
                "has_tools": has_py,
                "has_system_prompt": (sub / "prompt.md").exists(),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": 1,
            }
            added.append(name)

    # 2. битые записи (нет папки) → вон
    for name in list(registry.keys()):
        if not (SKILLS_DIR / name).exists():
            registry.pop(name, None)
            removed.append(name)

    # 3. проставить защиту базовым навыкам
    for name in PROTECTED_SKILLS:
        if name in registry and not registry[name].get("protected"):
            registry[name]["protected"] = True
            protected.append(name)

    # 4. протухшие временные навыки (не принятые retention-судьёй) → вычистить
    expired = []
    try:
        from omegaconf import OmegaConf

        ttl_days = float(OmegaConf.load("config.yml").get("skills", {}).get("temp_ttl_days", 7))
    except Exception:  # noqa: BLE001
        ttl_days = 7.0
    now = datetime.now()
    for name in list(registry.keys()):
        meta = registry[name]
        if not meta.get("temporary") or meta.get("protected"):
            continue
        try:
            age_days = (now - datetime.fromisoformat(meta.get("created_at", now.isoformat()))).days
        except ValueError:
            age_days = 0
        if age_days >= ttl_days:
            _save_registry(registry)  # registry мог измениться выше
            _delete_skill_impl(name, allow_protected=False)
            registry = _load_registry()
            expired.append(name)

    _save_registry(registry)
    return {"added": added, "removed": removed, "protected": protected, "expired_temp": expired}


def get_manager_tools() -> list:
    """Возвращает все management tools для передачи в агента."""
    from .openclaw_import import import_openclaw_skill_tool

    return [
        list_skills,
        read_skill,
        create_skill,
        update_skill_tools,
        delete_skill,
        load_skill_tools,
        get_skills_for_prompt,
        import_openclaw_skill_tool,
    ]


def get_all_loaded_skill_tools(names: Optional[list[str]] = None) -> list:
    """
    Загружает @tool функции из скиллов и возвращает их.

    names=None → все скиллы реестра (для прогрева при старте).
    names=[...] → ТОЛЬКО указанные скиллы — так в execution попадают лишь
    релевантные инструменты, а не весь реестр (анти-bloat контекста).
    """
    registry = _merged_registry()  # L4: глобальные + проектные навыки
    all_tools = []
    wanted = set(names) if names is not None else None

    for name, meta in registry.items():
        if wanted is not None and name not in wanted:
            continue
        if not meta.get("has_tools"):
            continue

        py_file = _skill_base(name) / f"{name}.py"
        if not py_file.exists():
            continue

        module, reason = _load_skill_module(name, py_file)  # гейт-перед-exec + кэш по mtime
        if module is None:
            print(f"[SkillManager] Skipped '{name}': {reason}")
            continue

        from src.runtime.hitl import needs_confirmation, wrap_with_confirmation

        guard = needs_confirmation(name)
        sandbox = _should_sandbox(name, meta)   # недоверенный → вызов в подпроцесс-песочнице (#2)
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if hasattr(obj, "name") and hasattr(obj, "invoke"):
                t = _sandbox_wrap(obj, name, py_file) if sandbox else obj
                all_tools.append(wrap_with_confirmation(t, name) if guard else t)

    return all_tools