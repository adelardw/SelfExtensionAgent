import ast
import json
import importlib.util
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool


SKILLS_DIR = Path("src/skills")
REGISTRY_FILE = SKILLS_DIR / "registry.json"


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
    REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _validate_python(code: str) -> tuple[bool, str]:
    """Проверяет синтаксис Python-кода БЕЗ выполнения."""
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"



@tool("list_skills")
def list_skills() -> str:
    """
    List all available skills with their descriptions and status.
    Use this FIRST to check what skills already exist before creating new ones.

    Returns:
        str: A formatted list of all registered skills.
    """
    registry = _load_registry()
    if not registry:
        return "No skills registered yet."

    lines = []
    for name, meta in registry.items():
        status = "ready" if meta.get("has_tools") else "description only"
        lines.append(
            f"• {name} [{status}] — {meta['description'][:100]}"
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
    skill_dir = SKILLS_DIR / name
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
            when this skill is active. Should explain HOW to use the tools,
            typical call patterns, and important constraints.

    Returns:
        str: Confirmation or error message.
    """
    _ensure_dirs()
    registry = _load_registry()

    if name in registry:
        return (
            f"Skill '{name}' already exists. "
            f"Use 'read_skill' to inspect it, or 'update_skill_tools' to modify."
        )

    if tool_code:
        is_valid, err = _validate_python(tool_code)
        if not is_valid:
            return f"Invalid Python code — {err}. Fix the code and try again."

    skill_dir = SKILLS_DIR / name
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
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "version": 1,
    }
    _save_registry(registry)

    result = f"Skill '{name}' created successfully."
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
    registry = _load_registry()
    if name not in registry:
        return f"Skill '{name}' does not exist. Create it first with 'create_skill'."

    is_valid, err = _validate_python(tool_code)
    if not is_valid:
        return f"Invalid Python code — {err}. Fix and retry."

    skill_file = SKILLS_DIR / name / f"{name}.py"

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
    _save_registry(registry)

    return f"Tools for skill '{name}' updated (v{registry[name]['version']})."


@tool("delete_skill")
def delete_skill(name: str) -> str:
    """
    Delete a skill entirely (description + tools + registry entry).
    Use with caution — this is irreversible.

    Args:
        name: The name of the skill to delete.

    Returns:
        str: Confirmation message.
    """
    import shutil

    registry = _load_registry()
    skill_dir = SKILLS_DIR / name

    if name not in registry and not skill_dir.exists():
        return f"Skill '{name}' not found."

    if skill_dir.exists():
        shutil.rmtree(skill_dir)

    registry.pop(name, None)
    _save_registry(registry)

    return f"Skill '{name}' has been deleted."


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
    py_file = SKILLS_DIR / name / f"{name}.py"
    if not py_file.exists():
        return f"Skill '{name}' has no tools file. Create tools first."

    code = py_file.read_text(encoding="utf-8")
    is_valid, err = _validate_python(code)
    if not is_valid:
        return f"Cannot load — invalid code: {err}"

    try:
        module_name = f"skills.{name}"

        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, str(py_file))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

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
        else:
            return (
                f"Skill '{name}' loaded but no @tool functions found. "
                f"Make sure functions are decorated with @tool."
            )

    except Exception as e:
        return f"Failed to load skill '{name}': {type(e).__name__}: {e}"


@tool("get_skills_for_prompt")
def get_skills_for_prompt() -> str:
    """
    Get all skill descriptions formatted for injection into the system prompt.
    Use this to understand what capabilities are currently available.

    Returns:
        str: Combined markdown descriptions of all registered skills.
    """
    registry = _load_registry()
    if not registry:
        return "No skills available."

    sections = []
    for name, meta in registry.items():
        md_file = SKILLS_DIR / name / f"{name}.md"
        if md_file.exists():
            content = md_file.read_text(encoding="utf-8")
            status = "tools ready" if meta.get("has_tools") else "no tools yet"
            sections.append(f"### Skill: {name} ({status})\n{content}")

    return "\n\n---\n\n".join(sections)




def get_skill_runtime_prompts(names: list[str]) -> str:
    """
    Возвращает объединённые системные промпты для указанных навыков.
    Используется при инъекции в execution-агента.
    Если у навыка нет prompt.md, фолбэчит на description (.md).
    """
    parts = []
    for name in names:
        prompt_file = SKILLS_DIR / name / "prompt.md"
        if prompt_file.exists():
            parts.append(
                f"[Навык: {name}]\n{prompt_file.read_text(encoding='utf-8')}"
            )
        else:
            md_file = SKILLS_DIR / name / f"{name}.md"
            if md_file.exists():
                parts.append(
                    f"[Навык: {name} (описание)]\n{md_file.read_text(encoding='utf-8')}"
                )
    return "\n\n---\n\n".join(parts) if parts else ""


def get_manager_tools() -> list:
    """Возвращает все management tools для передачи в агента."""
    return [
        list_skills,
        read_skill,
        create_skill,
        update_skill_tools,
        delete_skill,
        load_skill_tools,
        get_skills_for_prompt,
    ]


def get_all_loaded_skill_tools() -> list:
    """
    Сканирует все скиллы в реестре, загружает их модули
    и возвращает все найденные @tool функции.
    Вызывайте при старте агента чтобы подключить ранее созданные скиллы.
    """
    registry = _load_registry()
    all_tools = []

    for name, meta in registry.items():
        if not meta.get("has_tools"):
            continue

        py_file = SKILLS_DIR / name / f"{name}.py"
        if not py_file.exists():
            continue

        try:
            module_name = f"skills.{name}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, str(py_file))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if hasattr(obj, "name") and hasattr(obj, "invoke"):
                    all_tools.append(obj)

        except Exception as e:
            print(f"[SkillManager] Failed to load '{name}': {e}")

    return all_tools