"""
Импорт навыков OpenClaw (ClawHub) в нашу библиотеку навыков.

Формат OpenClaw: каталог со SKILL.md — YAML-frontmatter (name, description,
metadata.openclaw: {emoji, os, requires.bins, install[...]}) + markdown-инструкции,
которые учат агента работать с конкретным CLI (gh, memo и т.п.). Это
instruction-first навыки: исполняемая часть — системные бинарники.

Маппинг в наш формат (src/skills/<name>/):
  SKILL.md      — оригинал целиком (источник правды);
  <name>.md     — описание для реестра/селектора;
  prompt.md     — инструкции для инъекции исполнителю (обрезаны по бюджету);
  <name>.py     — авто-обёртка (НАШ доверенный шаблон, не LLM-код):
                    <name>_instructions() — полные инструкции,
                    <name>_run(command)   — запуск ТОЛЬКО разрешённых бинарников
                                            (requires.bins), timeout, dry-run.

Безопасность: сторонний скилл не получает произвольный шелл — только allowlist
своих бинарников; в реестре помечается imported=true, и hitl требует подтверждение
человека на его вызовы (при agent.require_confirmation). AST-гейт здесь не нужен:
шаблон пишем мы, а не LLM.

Источник: локальный каталог | git-URL (поддерживается github .../tree/<branch>/<subpath>).
CLI: python -m src.tools.openclaw_import <source> [имя_подкаталога_скилла]
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from langchain_core.tools import tool

from .skill_creation import SKILLS_DIR, _load_registry, _save_registry

PROMPT_BUDGET = 4000  # сколько символов инструкций инъектится исполнителю напрямую

_GH_TREE_RE = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+)(?:/tree/([\w.-]+)/(.+))?/?$")


def parse_skill_md(text: str) -> tuple[dict, str]:
    """SKILL.md → (frontmatter dict, markdown body)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), m.group(2)


def _sanitize(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower().replace("-", "_")).strip("_")
    return s or "imported_skill"


def _resolve_source(source: str, subpath: str = "") -> Path:
    """Локальный каталог как есть; git-URL — shallow clone во временный каталог."""
    p = Path(source).expanduser()
    if p.is_dir():
        return p / subpath if subpath else p
    m = _GH_TREE_RE.match(source)
    if m:
        owner, repo, _branch, tree_sub = m.groups()
        url = f"https://github.com/{owner}/{repo}"
        tmp = Path(tempfile.mkdtemp(prefix="openclaw_"))
        res = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(tmp)],
            capture_output=True, text=True, timeout=300,
        )
        if res.returncode != 0:
            raise ValueError(f"git clone failed: {res.stderr[-300:]}")
        sub = subpath or tree_sub or ""
        return tmp / sub if sub else tmp
    raise ValueError(f"Источник не найден: {source} (нужен локальный каталог или github-URL)")


def _find_skill_md(root: Path) -> Path:
    for cand in [root / "SKILL.md", root / "skill.md"]:
        if cand.exists():
            return cand
    hits = sorted(root.glob("*/SKILL.md"))
    if len(hits) == 1:
        return hits[0]
    if hits:
        names = ", ".join(h.parent.name for h in hits[:20])
        raise ValueError(f"В источнике несколько скиллов — укажи подкаталог. Найдены: {names}")
    raise ValueError(f"SKILL.md не найден в {root}")


# Шаблон обёртки. Плейсхолдеры через __NAME__-замену (не .format — в коде есть {}).
_WRAPPER_TEMPLATE = '''"""
Импортированный OpenClaw-скилл '__NAME__' (авто-обёртка, доверенный шаблон).
Источник: __SOURCE__
Instruction-first: сначала читай __NAME___instructions, затем работай через __NAME___run.
"""
import os
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

_DIR = Path(__file__).parent
_ALLOWED_BINS = __BINS__


@tool
def __NAME___instructions() -> str:
    """Полные инструкции импортированного OpenClaw-скилла '__NAME__'. ЧИТАЙ ПЕРВЫМ:
    тут субкоманды, флаги и примеры использования его CLI."""
    p = _DIR / "SKILL.md"
    return p.read_text(encoding="utf-8")[:12000] if p.exists() else "Инструкции не найдены."


@tool
def __NAME___run(command: str) -> str:
    """Выполнить CLI-команду скилла '__NAME__'. ПЕРВОЕ слово — один из разрешённых
    бинарников: __BINS_HUMAN__. Пример: '__EXAMPLE__'.
    Субкоманды и флаги — в __NAME___instructions."""
    parts = command.split()
    if not parts:
        return "Пустая команда."
    if parts[0] not in _ALLOWED_BINS:
        return f"Бинарник '{parts[0]}' не разрешён этому скиллу (allowlist: __BINS_HUMAN__)."
    if shutil.which(parts[0]) is None:
        return f"'{parts[0]}' не установлен. __INSTALL_HINT__"
    if os.getenv("AGENT_DRY_RUN") == "1":
        return f"[dry-run] {command}"
    try:
        r = subprocess.run(parts, capture_output=True, text=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    res = out + (("\\n[stderr] " + err) if err else "")
    return res[:4000] or f"(exit {r.returncode}, без вывода)"
'''


def _collect_bins(oc_meta: dict) -> list[str]:
    """Разрешённые бинарники = requires.bins ∪ все install[].bins (порядок, без дублей)."""
    out: list[str] = list((oc_meta.get("requires") or {}).get("bins") or [])
    for item in oc_meta.get("install") or []:
        out.extend(item.get("bins") or [])
    return list(dict.fromkeys(out))


def _install_hint(oc_meta: dict) -> str:
    hints = []
    for item in oc_meta.get("install") or []:
        kind = item.get("kind")
        if kind == "brew":
            hints.append(f"brew install {item.get('formula', '')}")
        elif kind == "apt":
            hints.append(f"apt install {item.get('package', '')}")
    return ("Установка: " + " | ".join(hints)) if hints else ""


def import_openclaw_skill(source: str, subpath: str = "", overwrite: bool = False) -> str:
    """Импортирует OpenClaw-скилл. Возвращает человекочитаемый отчёт."""
    root = _resolve_source(source, subpath)
    skill_md = _find_skill_md(root)
    raw = skill_md.read_text(encoding="utf-8")
    meta, body = parse_skill_md(raw)

    name = _sanitize(str(meta.get("name") or skill_md.parent.name))
    oc = (meta.get("metadata") or {}).get("openclaw") or {}
    bins = _collect_bins(oc)
    oses = oc.get("os") or []
    description = str(meta.get("description") or body.strip().split("\n")[0][:200])

    registry = _load_registry()
    if name in registry and not overwrite:
        return f"Навык '{name}' уже есть. Передай overwrite=True для замены."
    if registry.get(name, {}).get("protected"):
        return f"'{name}' — защищённый core-навык, импорт поверх него запрещён."

    notes = []
    if oses and "darwin" not in oses and sys.platform == "darwin":
        notes.append(f"⚠ скилл заявлен для os={oses}, текущая — darwin")
    missing = [b for b in bins if not shutil.which(b)]
    hint = _install_hint(oc)
    if missing:
        notes.append(f"⚠ не установлены бинарники: {', '.join(missing)}. {hint}".strip())

    dest = SKILLS_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SKILL.md").write_text(raw, encoding="utf-8")
    home = meta.get("homepage")
    (dest / f"{name}.md").write_text(
        f"{description}\n\n(Импортирован из OpenClaw: {source}"
        + (f", homepage: {home}" if home else "") + ")",
        encoding="utf-8",
    )
    prompt = (
        f"[Импортированный OpenClaw-скилл '{name}']\n"
        f"Работай через инструменты {name}_run (разрешены только: {', '.join(bins) or '—'}) "
        f"и {name}_instructions (полная справка).\n\n" + body.strip()[:PROMPT_BUDGET]
    )
    (dest / "prompt.md").write_text(prompt, encoding="utf-8")
    # сопутствующие файлы скилла (скрипты/ресурсы) — рядом, не как наш модуль
    for f in skill_md.parent.iterdir():
        if f.is_file() and f.name != "SKILL.md" and f.suffix != ".py":
            shutil.copy2(f, dest / f.name)

    example = f"{bins[0]} --help" if bins else ""
    wrapper = (
        _WRAPPER_TEMPLATE
        .replace("__NAME__", name)
        .replace("__SOURCE__", source)
        .replace("__BINS__", repr(set(bins)) if bins else "set()")
        .replace("__BINS_HUMAN__", ", ".join(bins) or "(нет — скилл только инструкции)")
        .replace("__EXAMPLE__", example)
        .replace("__INSTALL_HINT__", hint or "Установи бинарник вручную.")
    )
    (dest / f"{name}.py").write_text(wrapper, encoding="utf-8")

    registry[name] = {
        "description": description[:200],
        "has_tools": True,
        "has_system_prompt": True,
        "imported": True,
        "source": source,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "version": 1,
    }
    _save_registry(registry)

    report = f"OpenClaw-скилл '{name}' импортирован (CLI allowlist: {', '.join(bins) or 'нет'})."
    if notes:
        report += "\n" + "\n".join(notes)
    return report


@tool("import_openclaw_skill")
def import_openclaw_skill_tool(source: str, subpath: str = "") -> str:
    """
    Import an OpenClaw (ClawHub) skill into the skill library. Use ONLY when the user
    explicitly asks to install/import an OpenClaw skill.

    Args:
        source: Local directory with SKILL.md, or a GitHub URL
            (e.g. https://github.com/openclaw/openclaw/tree/main/skills/github).
        subpath: Optional sub-directory inside the source that holds the skill.

    Returns:
        str: Import report (created skill name, allowed CLIs, missing binaries).
    """
    try:
        return import_openclaw_skill(source, subpath)
    except Exception as e:  # noqa: BLE001
        return f"Импорт не удался: {type(e).__name__}: {e}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python -m src.tools.openclaw_import <источник> [подкаталог] [--overwrite]")
        sys.exit(1)
    args = [a for a in sys.argv[1:] if a != "--overwrite"]
    print(import_openclaw_skill(args[0], args[1] if len(args) > 1 else "",
                                overwrite="--overwrite" in sys.argv))
