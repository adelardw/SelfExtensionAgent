import ast
import json
import os
import importlib.util
import re
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


# ── SKILL.md frontmatter (agentskills.io-стиль, Hermes-порт) ──────────────────
# Создаваемые навыки получают YAML-frontmatter (name/description/when_to_use) → каталог
# переносим в другие harness'ы (Claude Code, OpenClaw и пр.) без конвертации; when_to_use —
# отдельное поле «когда использовать» для селектора. Чтение ТОЛЕРАНТНО: старые md без
# frontmatter работают как раньше.
_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """SKILL.md → (frontmatter dict, тело markdown). Нет frontmatter → ({}, текст как есть)."""
    m = _FM_RE.match(text or "")
    if not m:
        return {}, text or ""
    try:
        import yaml

        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:  # noqa: BLE001
        meta = {}
    return (meta if isinstance(meta, dict) else {}), m.group(2)


def _compose_skill_md(name: str, description: str, when_to_use: Optional[str]) -> str:
    """Обернуть описание навыка в frontmatter. Автор уже дал свой frontmatter → не дублируем."""
    if (description or "").lstrip().startswith("---"):
        return description
    first_line = next((ln.strip().lstrip("#").strip()
                       for ln in (description or "").splitlines() if ln.strip()), name)
    meta: dict = {"name": name, "description": first_line[:150]}
    if when_to_use and when_to_use.strip():
        meta["when_to_use"] = when_to_use.strip()
    try:
        import yaml

        fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    except Exception:  # noqa: BLE001
        return description
    return f"---\n{fm}\n---\n\n{description}"


def _skill_md_view(name: str, meta: dict) -> tuple[str, str]:
    """(when_to_use, тело md без frontmatter) навыка — для промпт-инъекции и ретривала."""
    md_file = _skill_base(name) / f"{name}.md"
    try:
        raw = md_file.read_text(encoding="utf-8")
    except OSError:
        return str(meta.get("when_to_use", "") or ""), str(meta.get("description", "") or "")
    fm, body = split_frontmatter(raw)
    wtu = str(fm.get("when_to_use", "") or meta.get("when_to_use", "") or "")
    return wtu, body.strip() or raw


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


# Имена навыков, созданных в текущем ПРОГОНЕ (структурный канал для create_skills_node,
# вместо хрупкого регэкспа по тексту сообщений агента). Скоуп по run_id (contextvar из
# run_context, наследуется вниз по asyncio): под конкурентным сервером и при фоновой
# дистилляции параллельные прогоны НЕ уводят имена друг у друга. Нет run-контекста
# (CLI/фон-поток) → общий ключ "_default" (прежнее поведение).
_session_created: dict[str, list[str]] = {}

try:  # per-run записи чистятся по завершении запроса (тот же механизм, что taint/research)
    from src.runtime.run_context import register_cleanup as _reg_cleanup

    _reg_cleanup(lambda rid: _session_created.pop(rid, None))
except Exception:  # noqa: BLE001
    pass


def _created_key() -> str:
    try:
        from src.runtime.run_context import current_run_id

        return current_run_id() or "_default"
    except Exception:  # noqa: BLE001
        return "_default"


def _note_created(name: str) -> None:
    _session_created.setdefault(_created_key(), []).append(name)


def pop_last_created() -> str:
    """Возвращает имя последнего навыка, созданного в ТЕКУЩЕМ прогоне (и забывает его)."""
    lst = _session_created.get(_created_key())
    return lst.pop() if lst else ""


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


# ── жизненный цикл: usage-статистика навыков (Hermes-порт, аналог Curator) ─────
# Каждый прогон, где навык был выбран в execution, пишет uses/wins/last_used_at в реестр.
# Куратор в sync_registry() по этим числам вычищает систематически проигрывающие
# сгенерированные навыки (protected/imported не трогает). В eval-режиме не пишем
# (бенч-исходы не должны приговаривать навыки боевой библиотеки).

def record_skill_usage(names: list[str], win: bool) -> None:
    """Фиксирует факт использования навыков в прогоне (+исход). Ошибки глотаем: статистика
    не должна ронять reflect. No-op в eval И под pytest: бенч-исходы и юнит-тесты, гоняющие
    act/reflect с БОЕВЫМ реестром, не должны красить статистику библиотеки (вскрыто живым
    прогоном сьюта: тесты act записали uses/wins реальным навыкам)."""
    if os.getenv("AGENT_EVAL_MODE") == "1":
        return
    # Под pytest — no-op (тесты act/reflect гоняют БОЕВОЙ реестр), кроме явного opt-in
    # тестов самой статистики (они работают с временным реестром).
    if os.getenv("PYTEST_CURRENT_TEST") and os.getenv("AGENT_USAGE_TRACKING_IN_TESTS") != "1":
        return
    for name in names or []:
        try:
            path = _registry_path_for(name)
            registry = _load_reg_at(path)
            meta = registry.get(name)
            if meta is None:
                continue
            meta["uses"] = int(meta.get("uses", 0)) + 1
            meta["wins"] = int(meta.get("wins", 0)) + (1 if win else 0)
            meta["last_used_at"] = datetime.now().isoformat()
            # Retention ДЕЛОМ: temporary-навык (в т.ч. дистиллированный из траектории),
            # принёсший победу в реальном прогоне, принимается в библиотеку — иначе его
            # снесёт TTL-чистка, хотя он доказал полезность.
            if win:
                meta.pop("temporary", None)
            _save_reg_at(path, registry)
        except Exception:  # noqa: BLE001
            pass


def skill_usage_stats() -> dict[str, dict]:
    """{name: {uses, wins, last_used_at}} по объединённому реестру (для CLI/бенча/куратора)."""
    out = {}
    for name, meta in _merged_registry().items():
        out[name] = {
            "uses": int(meta.get("uses", 0)),
            "wins": int(meta.get("wins", 0)),
            "last_used_at": meta.get("last_used_at", ""),
        }
    return out



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
        uses = int(meta.get("uses", 0))
        stats = f" ({int(meta.get('wins', 0))}/{uses} wins)" if uses else ""
        lines.append(
            f"• {name} [{status}]{lock}{scope}{temp}{imp}{stats} — {meta['description'][:100]}"
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
    when_to_use: Optional[str] = None,
) -> str:
    """
    Create a new skill with description, system prompt, and optionally tool code.
    The skill's .md is saved with agentskills.io-style YAML frontmatter (portable format).

    Args:
        name: Short snake_case name for the skill (e.g. 'web_search', 'data_analysis').
        description: Markdown description of WHEN and HOW to use this skill.
            Include: purpose, triggers, inputs/outputs, examples.
        when_to_use: One-two sentences: in which situations/queries this skill should be
            picked. Used by the skill selector — write concrete trigger phrases.
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

    (skill_dir / f"{name}.md").write_text(
        _compose_skill_md(name, description, when_to_use), encoding="utf-8")

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
    if when_to_use and when_to_use.strip():
        registry[name]["when_to_use"] = when_to_use.strip()[:300]
    _save_reg_at(reg_path, registry)
    _note_created(name)

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
        if not (_skill_base(name) / f"{name}.md").exists():
            continue
        wtu, body = _skill_md_view(name, meta)  # frontmatter в промпт не течёт
        status = "tools ready" if meta.get("has_tools") else "no tools yet"
        head = f"### Skill: {name} ({status})"
        if wtu:
            head += f"\nКогда использовать: {wtu}"
        sections.append(f"{head}\n{body}")

    return "\n\n---\n\n".join(sections)


# Порог: пока навыков мало — показываем все (дёшево). При росте библиотеки селектор
# захлёбывается полным списком → включаем ToolSearch (retrieval топ-релевантных).
TOOLSEARCH_THRESHOLD = 12
TOOLSEARCH_TOP = 8


# ── гибридный ранкер навыков (Hermes-порт: масштабирование выбора) ─────────────
# BM25 по md + (при включённых memory.embeddings) косинус по кэшированным векторам,
# слитые RRF. Векторы считаются лениво и кэшируются на диске с инвалидацией по mtime
# md — на сотнях навыков это один embed-вызов на ЗАПРОС (или ноль, если qvec передан
# из recall), а не на навык. Эмбеддер выключен/упал → чистый BM25, как раньше.
_EMB_CACHE_LOCK = threading.Lock()
_skill_embedder_inst = None


def _skill_embedder():
    global _skill_embedder_inst
    if _skill_embedder_inst is None:
        try:
            from omegaconf import OmegaConf

            from src.memory.embedder import build_embedder

            mc = OmegaConf.load("config.yml").get("memory", {})
            _skill_embedder_inst = build_embedder(
                bool(mc.get("embeddings", False)), mc.get("embedding_model"))
        except Exception:  # noqa: BLE001
            from src.memory.embedder import NullEmbedder

            _skill_embedder_inst = NullEmbedder()
    return _skill_embedder_inst


# Memo распарсенного кэша векторов: файл ~30-40КБ НА НАВЫК → перечитывать/парсить JSON на
# КАЖДЫЙ ранк-запрос при сотнях навыков стало бы мегабайтами на запрос. Держим разобранный
# dict в памяти, перечитываем только если файл на диске сменился (mtime).
_emb_memo: dict = {"mtime": None, "data": {}}


def _cached_skill_vecs(names: list[str], docs: list[str], emb) -> list[Optional[list]]:
    """Векторы документов навыков из дискового кэша (SKILLS_DIR/.emb_cache.json),
    инвалидация по mtime md; недостающие докачиваются эмбеддером и дописываются.
    Мёртвые ключи (удалённые навыки) вычищаются при записи."""
    cache_path = SKILLS_DIR / ".emb_cache.json"
    try:
        disk_mtime = cache_path.stat().st_mtime if cache_path.exists() else None
    except OSError:
        disk_mtime = None
    if _emb_memo["mtime"] == disk_mtime and disk_mtime is not None:
        cache = _emb_memo["data"]
    else:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        except Exception:  # noqa: BLE001
            cache = {}
        _emb_memo.update(mtime=disk_mtime, data=cache)
    vecs: list[Optional[list]] = []
    dirty = False
    for name, doc in zip(names, docs):
        try:
            mtime = (_skill_base(name) / f"{name}.md").stat().st_mtime
        except OSError:
            mtime = 0.0
        entry = cache.get(name)
        if entry and entry.get("mtime") == mtime and entry.get("vec"):
            vecs.append(entry["vec"])
            continue
        vec = emb.embed(doc[:4000])
        vecs.append(vec)
        if vec:
            cache[name] = {"mtime": mtime, "vec": vec}
            dirty = True
    if dirty:
        try:
            alive = set(_merged_registry())
            for dead in [k for k in cache if k not in alive]:
                cache.pop(dead, None)  # удалённые навыки не пухнут в кэше вечно
            with _EMB_CACHE_LOCK:
                _atomic_write_json(cache_path, cache)
                _emb_memo.update(mtime=cache_path.stat().st_mtime, data=cache)
        except Exception:  # noqa: BLE001
            pass
    return vecs


def rank_skill_docs(docs: list[str], query: str, top: int,
                    names: Optional[list[str]] = None,
                    qvec: Optional[list] = None) -> list[int]:
    """Индексы топ-`top` документов навыков: RRF(BM25, cosine-по-эмбеддингам).
    Без эмбеддера (или без names для кэша) — чистый BM25 (прежнее поведение)."""
    from src.search.retrieval import bm25_rank

    bm_idx = bm25_rank(docs, query, top)
    emb = _skill_embedder()
    if not getattr(emb, "enabled", False) or names is None or len(names) != len(docs):
        return bm_idx
    try:
        from src.memory.embedder import cosine

        qv = qvec or emb.embed(query)
        if not qv:
            return bm_idx
        vecs = _cached_skill_vecs(names, docs, emb)
        sims = [(i, cosine(qv, v)) for i, v in enumerate(vecs) if v]
        emb_idx = [i for i, s in sorted(sims, key=lambda p: -p[1]) if s > 0][:top]
    except Exception:  # noqa: BLE001
        return bm_idx
    if not emb_idx:
        return bm_idx
    # RRF: устойчивое слияние двух ранжировок без калибровки скорингов
    score: dict[int, float] = {}
    for lst in (bm_idx, emb_idx):
        for pos, i in enumerate(lst):
            score[i] = score.get(i, 0.0) + 1.0 / (60 + pos)
    return sorted(score, key=lambda i: -score[i])[:top]


def get_relevant_skills_for_prompt(query: str, top: int = TOOLSEARCH_TOP) -> str:
    """
    ToolSearch: BM25-retrieval НАВЫКОВ по запросу (вместо показа ВСЕХ селектору). Масштабирует
    выбор инструментов при росте библиотеки. Reuse канонического ранкера (src.retrieval).
    Если навыков мало (< порога) или нет совпадений — фолбэк на полный список.
    """
    registry = _merged_registry()  # L4: глобальные + проектные навыки
    if not registry or len(registry) < TOOLSEARCH_THRESHOLD:
        return get_skills_for_prompt.invoke({})  # это @tool, не функция

    names, docs, sections = [], [], []
    for name, meta in registry.items():
        if not (_skill_base(name) / f"{name}.md").exists():
            continue
        wtu, body = _skill_md_view(name, meta)  # frontmatter в промпт не течёт
        status = "tools ready" if meta.get("has_tools") else "no tools yet"
        names.append(name)
        docs.append(f"{name} {wtu} {body}")  # when_to_use — сигнал ретривала (Hermes-порт)
        head = f"### Skill: {name} ({status})"
        if wtu:
            head += f"\nКогда использовать: {wtu}"
        sections.append(f"{head}\n{body}")

    idx = rank_skill_docs(docs, query, top, names=names)
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
                _, body = split_frontmatter(md_file.read_text(encoding="utf-8"))
                parts.append(f"[Навык: {name} (описание)]\n{body}")
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
            desc = split_frontmatter(md.read_text(encoding="utf-8"))[1].strip()[:200] if md.exists() else ""
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

    # 5. КУРАТОР (Hermes-порт): сгенерированный навык, который систематически проигрывает
    # (достаточно использований, а побед почти нет), — мусор в реестре: жрёт контекст селектора
    # и подсовывается заново. Чистим по СТАТИСТИКЕ (uses/wins из record_skill_usage), а не по
    # слепому TTL. Protected/imported не трогаем; порог консервативный (config skills.curator_*).
    try:
        from omegaconf import OmegaConf

        _sk = OmegaConf.load("config.yml").get("skills", {})
        min_uses = int(_sk.get("curator_min_uses", 5))
        win_floor = float(_sk.get("curator_win_floor", 0.2))
    except Exception:  # noqa: BLE001
        min_uses, win_floor = 5, 0.2
    curated = []
    for name in list(registry.keys()):
        meta = registry[name]
        if meta.get("protected") or meta.get("imported") or name in PROTECTED_SKILLS:
            continue
        uses = int(meta.get("uses", 0))
        wins = int(meta.get("wins", 0))
        if uses >= min_uses and (wins / uses) < win_floor:
            _save_registry(registry)
            _delete_skill_impl(name, allow_protected=False)
            registry = _load_registry()
            curated.append(name)

    _save_registry(registry)
    return {"added": added, "removed": removed, "protected": protected,
            "expired_temp": expired, "curated_out": curated}


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