"""
`.sea/` — рабочий каталог проекта для CLI-агента `sea` (L1). Как `.git/` у git: per-project
локальное состояние. Создаётся `sea init`. Хранит историю/конфиг/лог решений (accept/reject).

АДДИТИВНО И БЕЗОПАСНО: пока `.sea/` нет — `log_decision()` это no-op, поведение агента не
меняется. Каталог cwd-relative (project-rooted), как и остальной runtime-стейт (data/, config.local).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

SEA_DIR = Path(".sea")
_HISTORY = SEA_DIR / "history"
_DECISIONS = _HISTORY / "decisions.jsonl"

# Минимальные стартовые конвенции (создаются `sea init`, если файла ещё нет).
_SEA_MD = """# SEA.md — инструкции проекта для агента

Агент `sea` держит этот файл в контексте каждого запроса (как CLAUDE.md). Пиши тут правила
проекта: кто пользователь, стиль ответов, что делать/не делать.

- (пример) Отвечай развёрнуто, со ссылками на источники.
"""

_MEMORY_MD = """# Project Memory Index

Курируемая память проекта. Агент подмешивает релевантные заметки на recall; типизированные
заметки пишет в data/project_memory/ (type: user|feedback|project|reference) и добавляет сюда
строку-указатель. Можно вести и вручную.
"""

_MCP_MD = """# MCP.md — пользовательский реестр MCP-серверов

Поля (как SKILL.md): name, transport(stdio|sse|streamable_http), command, args, url, keywords, trusted.

```yaml
servers: []
```
"""

_CONVENTIONS = {"SEA.md": _SEA_MD, "MEMORY.md": _MEMORY_MD, "MCP.md": _MCP_MD}

# Папки, которые НЕ сканируем (мусор/большие).
_IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "data", "dist",
                "build", ".sea", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea",
                ".vscode", "target", "vendor", ".next", "coverage", "logs", "site-packages",
                ".egg-info", "htmlcov", ".tox"}
_LANG = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
         ".jsx": "JavaScript", ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby",
         ".c": "C", ".cpp": "C++", ".h": "C/C++", ".hpp": "C++", ".cs": "C#", ".php": "PHP",
         ".swift": "Swift", ".kt": "Kotlin", ".scala": "Scala", ".sh": "Shell", ".sql": "SQL",
         ".html": "HTML", ".css": "CSS", ".vue": "Vue", ".md": "Markdown"}
_MANIFESTS = ("pyproject.toml", "setup.py", "requirements.txt", "package.json", "go.mod",
              "Cargo.toml", "pom.xml", "build.gradle", "Gemfile", "composer.json",
              "Makefile", "Dockerfile", "docker-compose.yml", "README.md", "README.rst")


def _detect_commands(base: Path) -> list[str]:
    """Команды запуска/тестов из манифестов (детерминированно, без исполнения)."""
    cmds: list[str] = []
    py = base / "pyproject.toml"
    if py.is_file():
        try:
            import tomllib
            data = tomllib.loads(py.read_text(encoding="utf-8"))
            for name in (data.get("project", {}).get("scripts") or {}):
                cmds.append(f"`{name}` (cli-команда из pyproject)")
            deps = " ".join(data.get("project", {}).get("dependencies", []) or [])
            if "pytest" in deps or (base / "uv.lock").exists():
                cmds.append("тесты: `pytest` (или `uv run pytest`)")
        except Exception:  # noqa: BLE001
            pass
    pkg = base / "package.json"
    if pkg.is_file():
        try:
            import json as _json
            scripts = (_json.loads(pkg.read_text(encoding="utf-8")).get("scripts") or {})
            for k in list(scripts)[:6]:
                cmds.append(f"`npm run {k}`")
        except Exception:  # noqa: BLE001
            pass
    mk = base / "Makefile"
    if mk.is_file():
        try:
            for line in mk.read_text(encoding="utf-8").splitlines():
                m = line.split(":", 1)[0].strip()
                if m and " " not in m and not line.startswith("\t") and m.isidentifier():
                    cmds.append(f"`make {m}`")
        except Exception:  # noqa: BLE001
            pass
    return cmds[:8]


def scan_repo(base: Path | None = None) -> str:
    """Детерминированная карта репозитория в markdown (для стартовой SEA.md). Без LLM/чтения
    содержимого — только структура/расширения/манифесты. Безопасно: ошибки → краткий фолбэк."""
    base = base or root()
    try:
        lang_counts: dict[str, int] = {}
        nfiles = 0
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames
                           if d not in _IGNORE_DIRS and not d.startswith(".") and not d.endswith(".egg-info")]
            for fn in filenames:
                lang = _LANG.get(Path(fn).suffix.lower())
                if lang:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                nfiles += 1
            if nfiles > 50000:
                break
        langs = sorted(lang_counts.items(), key=lambda x: -x[1])[:6]
        top_dirs = sorted(p.name for p in base.iterdir()
                          if p.is_dir() and p.name not in _IGNORE_DIRS and not p.name.startswith("."))
        manifests = [f for f in _MANIFESTS if (base / f).is_file()]
        cmds = _detect_commands(base)

        out = ["# SEA.md — инструкции проекта (стартовая карта от `sea init`)", "",
               "Автоматический обзор репозитория. Агент держит этот файл в контексте каждого "
               "запроса. Дополни/поправь инструкции проекта внизу.", ""]
        out.append("## Стек")
        out.append(", ".join(f"{l} ({n})" for l, n in langs) if langs else "(языков не распознано)")
        out += ["", "## Структура (верхний уровень)",
                " · ".join(f"`{d}/`" for d in top_dirs) or "(нет подпапок)"]
        out += ["", "## Ключевые файлы", ", ".join(f"`{m}`" for m in manifests) or "(манифестов не найдено)"]
        out += ["", "## Команды"]
        out += [("\n".join(f"- {c}" for c in cmds)) if cmds else "- (не распознаны автоматически)"]
        out += ["", "## Инструкции проекта",
                "(допиши здесь правила: стиль ответов, что делать/не делать, контекст проекта)", ""]
        return "\n".join(out)
    except Exception as e:  # noqa: BLE001
        return ("# SEA.md — инструкции проекта\n\n(авто-скан не удался: "
                f"{type(e).__name__}). Опиши проект и правила вручную.\n")


def root() -> Path:
    return Path(os.getenv("AGENT_PROJECT_ROOT") or Path.cwd())


def sea_dir() -> Path:
    return root() / SEA_DIR


def is_initialized() -> bool:
    return sea_dir().is_dir()


def init(scaffold_conventions: bool = True) -> list[str]:
    """Создать `.sea/` (+ history) и, опционально, стартовые SEA.md/MEMORY.md/MCP.md (если их нет).
    Возвращает список созданных путей (для вывода в CLI). Идемпотентно — не перезаписывает."""
    created: list[str] = []
    base = root()
    d = base / SEA_DIR
    for sub in (d, base / _HISTORY):
        if not sub.exists():
            sub.mkdir(parents=True, exist_ok=True)
            created.append(str(sub.relative_to(base)))
    readme = d / "README.md"
    if not readme.exists():
        readme.write_text(
            "Рабочий каталог проекта для `sea`. Здесь: history/ (чаты, лог решений accept/reject), "
            "локальный конфиг. Можно добавить в .gitignore.\n", encoding="utf-8")
        created.append(str(readme.relative_to(base)))
    if scaffold_conventions:
        for name, body in _CONVENTIONS.items():
            p = base / name
            if not p.exists():
                # SEA.md = АВТО-КАРТА репозитория (скан), чтобы агент знал проект с 1-го запроса;
                # MEMORY.md/MCP.md — пустые шаблоны.
                content = scan_repo(base) if name == "SEA.md" else body
                p.write_text(content, encoding="utf-8")
                created.append(name)
    return created


def log_decision(action: str, approved: bool, kind: str = "", note: str = "") -> None:
    """Записать решение accept/reject во временный реестр проекта. NO-OP пока `.sea/` нет
    (аддитивно: без init поведение не меняется). Ошибки записи не роняют прогон."""
    if not is_initialized():
        return
    try:
        rec = {"ts": time.time(), "action": action[:300], "approved": bool(approved),
               "kind": kind, "note": note[:300]}
        path = root() / _DECISIONS
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def decisions(limit: int = 50) -> list[dict]:
    """Последние решения из лога (для /history или ревью). [] если нет."""
    path = root() / _DECISIONS
    if not path.exists():
        return []
    try:
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return rows[-limit:]
    except Exception:  # noqa: BLE001
        return []
