"""
Навык `code` (L2): обзор и правка кодовой базы. Порт паттернов Glob/Grep/FileRead/FileEdit/Bash
из ~/Desktop/backup (Claude Code).

Разрешения (через L1-HITL, зависит от work_mode):
  • READ-ONLY (без подтверждения): glob_files, grep_repo, list_tree, read_lines — в
    hitl._DEFAULT_READONLY, поэтому исполняются свободно.
  • SIDE-EFFECT (подтверждение + зависит от мода): edit_file, run_bash. Навык `code` в
    config skills.confirm → эти тулзы оборачиваются: plan → блок, manual → спрос, auto → выполнить.
    run_bash дополнительно уважает AGENT_DRY_RUN (второй предохранитель).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool

_IGNORE = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build",
           ".sea", ".mypy_cache", ".pytest_cache", ".ruff_cache", "target", "vendor", ".next",
           ".idea", ".vscode", "coverage", "htmlcov", ".tox"}
_MAXOUT = 6000        # потолок вывода тула (анти-раздувание контекста)
_MAXFILE = 200_000    # потолок чтения файла


def _ignored(p: Path) -> bool:
    return any(part in _IGNORE for part in p.parts)


# ── Скоуп файловых тулз к корню проекта + денилист секрет-файлов ──────────────────────
# read-тулзы (glob/grep/tree/read_lines) — в hitl._DEFAULT_READONLY → HITL НЕ зовётся НИКОГДА.
# Без скоупа заинъекченный веб-контентом агент прочёл бы ~/.ssh/.env/credentials и слил на
# внешний хост (browse режет только internal/SSRF, не external) — эксфильтрация мимо всех гейтов
# (баг ревью: read-близнец закрытого run_bash-RCE). Барьер: путь УДЕРЖИВАЕТСЯ в корне проекта
# (.resolve() ловит и symlink-escape), а секрет-файлы не читаются даже ВНУТРИ репо (.env в корне).
def _project_root() -> Path:
    """Корень проекта (как context_files/skill_creation): AGENT_PROJECT_ROOT или cwd."""
    return Path(os.getenv("AGENT_PROJECT_ROOT") or Path.cwd()).resolve()


_SECRET_NAMES = {".env", ".netrc", ".pgpass", ".htpasswd", "credentials"}
_SECRET_SUBSTR = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", ".pem", ".key", ".keystore",
                  ".pfx", ".p12", ".env.", "secret", ".npmrc", ".pypirc", "credential", ".aws")
# Исключения секрет-файлов для ripgrep (быстрый путь grep_repo).
_RG_SECRET_EXCLUDES = ("!.env", "!*.env", "!*.env.*", "!*.pem", "!*.key", "!id_rsa*",
                       "!id_ed25519*", "!*.keystore", "!*.pfx", "!*.p12", "!.npmrc",
                       "!.pypirc", "!.netrc", "!.pgpass", "!*credential*", "!*secret*")


def _is_secret_file(p: Path) -> bool:
    n = p.name.lower()
    return n in _SECRET_NAMES or any(s in n for s in _SECRET_SUBSTR)


def _within_root(p: Path) -> bool:
    """p (после resolve) лежит в корне проекта?"""
    root = _project_root()
    try:
        rp = p.resolve()
    except (OSError, RuntimeError):
        return False
    return rp == root or root in rp.parents


def _safe_path(path: str) -> tuple[Path | None, str]:
    """Резолвит путь и удерживает его ВНУТРИ корня проекта. (resolved | None, error)."""
    root = _project_root()
    try:
        cand = Path(path)
        rp = (cand if cand.is_absolute() else root / cand).resolve()
    except (OSError, ValueError, RuntimeError) as e:
        return None, f"плохой путь: {e}"
    if rp != root and root not in rp.parents:
        return None, (f"путь вне проекта запрещён (тулзы code ограничены корнем "
                      f"{root.name}/): {path}")
    return rp, ""


@tool
def glob_files(pattern: str, path: str = ".") -> str:
    """Найти файлы по glob-паттерну (напр. '**/*.py', 'src/**/*.ts') в каталоге path.
    Read-only, в пределах корня проекта. Возвращает список путей (до 200)."""
    base, err = _safe_path(path)
    if err:
        return err
    if not base.exists():
        return f"путь не найден: {path}"
    out = []
    try:
        for p in base.glob(pattern):
            # _within_root: паттерн с '..' не должен вытащить файл за корень.
            if p.is_file() and not _ignored(p) and not _is_secret_file(p) and _within_root(p):
                out.append(str(p))
                if len(out) >= 200:
                    break
    except Exception as e:  # noqa: BLE001
        return f"ошибка glob: {e}"
    return "\n".join(sorted(out)) or "(ничего не найдено)"


@tool
def grep_repo(pattern: str, path: str = ".", file_glob: str = "") -> str:
    """Искать РЕГЭКСП по содержимому файлов рекурсивно (как ripgrep). Read-only, в пределах
    корня проекта. Возвращает строки 'file:line: текст' (до лимита). file_glob — опц. фильтр имён."""
    base, err = _safe_path(path)
    if err:
        return err
    # Быстрый путь: ripgrep, если установлен. Секрет-файлы исключаем глобами (-g '!...').
    try:
        cmd = ["rg", "-n", "--no-heading", "-S", "--max-columns", "200"]
        for g in _RG_SECRET_EXCLUDES:
            cmd += ["-g", g]
        if file_glob:
            cmd += ["-g", file_glob]
        cmd += [pattern, str(base)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode in (0, 1):  # 1 = нет совпадений (не ошибка)
            return (r.stdout[:_MAXOUT].strip() or "(совпадений нет)") + (
                "\n…(обрезано)" if len(r.stdout) > _MAXOUT else "")
    except Exception:  # noqa: BLE001 — нет rg → python-фолбэк
        pass
    # Фолбэк: python os.walk + re.
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"плохой регэксп: {e}"
    hits, total = [], 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE and not d.startswith(".")]
        for fn in filenames:
            if file_glob and not Path(fn).match(file_glob):
                continue
            fp = Path(dirpath) / fn
            if _is_secret_file(fp):  # секрет-файл не читаем даже внутри репо
                continue
            try:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append(f"{fp}:{i}: {line.rstrip()[:200]}")
                            total += 1
                            if total >= 100:
                                return "\n".join(hits) + "\n…(обрезано на 100)"
            except OSError:
                continue
    return "\n".join(hits) or "(совпадений нет)"


@tool
def list_tree(path: str = ".", depth: int = 2) -> str:
    """Дерево каталога до глубины depth (read-only, в пределах корня проекта). Пропускает
    мусорные/большие папки и секрет-файлы."""
    base, err = _safe_path(path)
    if err:
        return err
    if not base.exists():
        return f"путь не найден: {path}"
    lines, n = [], 0
    base_depth = len(base.parts)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in sorted(dirnames) if d not in _IGNORE and not d.startswith(".")]
        d = Path(dirpath)
        lvl = len(d.parts) - base_depth
        if lvl > depth:
            dirnames[:] = []
            continue
        indent = "  " * lvl
        lines.append(f"{indent}{d.name}/")
        for fn in sorted(filenames)[:40]:
            if _is_secret_file(Path(fn)):  # не светим .env/ключи в дереве
                continue
            lines.append(f"{indent}  {fn}")
        n += 1
        if n > 400:
            lines.append("…(обрезано)")
            break
    return "\n".join(lines)


@tool
def read_lines(file_path: str, start: int = 1, end: int = 200) -> str:
    """Прочитать файл с НОМЕРАМИ строк, диапазон [start, end] (read-only, в пределах корня
    проекта; секрет-файлы недоступны). Для точечной правки."""
    p, err = _safe_path(file_path)
    if err:
        return err
    if _is_secret_file(p):
        return ("доступ к секрет-файлам запрещён (.env/ключи/credentials не читаются — "
                "анти-эксфильтрация)")
    if not p.is_file():
        return f"файл не найден: {file_path}"
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")[:_MAXFILE]
    except OSError as e:
        return f"ошибка чтения: {e}"
    lines = text.splitlines()
    start = max(1, start)
    chunk = lines[start - 1:end]
    return "\n".join(f"{start + i}\t{ln}" for i, ln in enumerate(chunk)) or "(пусто/вне диапазона)"


@tool
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """ТОЧЕЧНАЯ правка: заменить old_string на new_string в файле (как FileEditTool).
    old_string должен встречаться РОВНО ОДИН раз (иначе откажет — добавь контекст). SIDE-EFFECT:
    проходит через подтверждение и зависит от режима (plan → не исполнится). Запись — только в
    пределах корня проекта, секрет-файлы недоступны."""
    p, err = _safe_path(file_path)
    if err:
        return err
    if _is_secret_file(p):
        return "правка секрет-файлов запрещена (.env/ключи/credentials защищены)"
    if not p.is_file():
        return f"файл не найден: {file_path}"
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        return f"ошибка чтения: {e}"
    cnt = text.count(old_string)
    if cnt == 0:
        return "old_string не найден — скопируй точный фрагмент (с отступами) через read_lines."
    if cnt > 1:
        return f"old_string встречается {cnt} раз — добавь окружающий контекст, чтобы он был уникален."
    try:
        p.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    except OSError as e:
        return f"ошибка записи: {e}"
    return f"OK: правка применена в {file_path}"


@tool
def run_bash(command: str, cwd: str = "", timeout: int = 60) -> str:
    """Выполнить shell-команду (как BashTool). SIDE-EFFECT: требует подтверждения и ЗАВИСИТ от
    режима (plan → блок, manual → спрос, auto → выполнить). При AGENT_DRY_RUN — не исполняет."""
    if os.getenv("AGENT_DRY_RUN") == "1":
        return f"[dry-run] не исполняю: {command}"
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True,
                           timeout=min(int(timeout), 300), cwd=(cwd or None))
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        tag = "" if r.returncode == 0 else f"[exit {r.returncode}] "
        return tag + (out[:_MAXOUT].strip() or "(пустой вывод)") + ("\n…(обрезано)" if len(out) > _MAXOUT else "")
    except subprocess.TimeoutExpired:
        return f"таймаут ({timeout}с): {command}"
    except Exception as e:  # noqa: BLE001
        return f"ошибка запуска: {e}"
