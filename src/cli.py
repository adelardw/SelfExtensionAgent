"""
Лёгкая точка входа консольной команды `sea` (см. pyproject [project.scripts]).

Принцип: `--version`/`--help` отвечают МГНОВЕННО и БЕЗ импорта агента (main.py греет
langchain/модели/навыки на уровне модуля — это пара секунд и требует config.yml). Тяжёлый
путь (REPL / one-shot прогон) импортирует main ЛЕНИВО, только когда реально нужен.

Использование:
  sea                     интерактивный REPL (как `claude`)
  sea "вопрос/задача"     one-shot: выполнить и выйти (код возврата 0/1)
  sea --auto "задача"     one-shot без HITL-подтверждений (авто-режим)
  sea init                создать .sea/ + стартовые SEA.md/MEMORY.md/MCP.md
  sea --version | -V      версия
  sea --help    | -h      эта справка
"""
from __future__ import annotations

import sys

_USAGE = """sea — self-extension-agent (CLI)

Использование:
  sea                     запустить интерактивный REPL (диалог в терминале)
  sea "<задача>"          выполнить одну задачу и выйти (one-shot)
  sea --auto "<задача>"   one-shot без подтверждений (автономно)
  sea init                создать .sea/ (история, лог решений) + стартовые SEA.md/MEMORY.md/MCP.md
  sea --version, -V       показать версию
  sea --help, -h          показать эту справку

В REPL доступны слэш-команды: /model /facts /goal /usage /diagnose /kb /chats и др.
Команда работает из каталога проекта (рядом должны быть config.yml и .env с ключом провайдера)."""


def _version() -> str:
    """Версия из метаданных установленного пакета; фолбэк — чтение pyproject; иначе '?'."""
    try:
        from importlib.metadata import version
        return version("self-extension-agent")
    except Exception:  # noqa: BLE001
        pass
    try:
        import tomllib
        from pathlib import Path
        for base in (Path(__file__).resolve().parent.parent, Path.cwd()):
            p = base / "pyproject.toml"
            if p.exists():
                return tomllib.loads(p.read_text(encoding="utf-8"))["project"]["version"]
    except Exception:  # noqa: BLE001
        pass
    return "?"


def main_cli() -> None:
    """Точка входа `sea`. Лёгкие флаги — здесь; всё остальное → ленивый main._cli_entry."""
    args = sys.argv[1:]
    first = args[0] if args else ""

    if first in ("--version", "-V", "version"):
        print(f"sea {_version()}")
        return
    if first in ("--help", "-h", "help"):
        print(_USAGE)
        return
    if first == "init":
        # Лёгкая команда: создать .sea/ + стартовые конвенции. Без импорта агента.
        from .sea_workspace import init
        created = init()
        if created:
            print("sea init — создано:\n  " + "\n  ".join(created))
        else:
            print("sea init — уже инициализировано (ничего не перезаписано).")
        return
    # ЕДИНСТВЕННЫЙ интерактивный интерфейс — full-screen TUI (Textual): `sea` без аргументов
    # ИЛИ `sea --tui`/`tui`. Старый line-mode REPL УДАЛЁН (TUI его полностью заменил).
    if first in ("--tui", "tui") or not first:
        from .tui import run_tui
        run_tui()
        return

    # Тяжёлый путь — ТОЛЬКО one-shot `sea "задача"` [--auto] (для скриптов/cron, без интерактива).
    # Импортируем агента ЛЕНИВО — только теперь.
    # ВАЖНО: грузим ЖИВОЙ main.py из каталога проекта (cwd), а не устаревшую КОПИЮ, которую
    # editable-install кладёт в site-packages (main.py — root-модуль → копируется как обычный
    # файл, НЕ editable; из-за этого `sea` грузил старый баннер и не видел /init — баг ревью).
    # cwd в начало sys.path → живой файл выигрывает у копии. (Каноничнее — перенести main.py в
    # src/repl.py, чтобы он был частью editable-пакета; см. develop.md TODO дистрибуции.)
    import os as _os

    _root = _os.getcwd()
    while _root in sys.path:
        sys.path.remove(_root)
    sys.path.insert(0, _root)
    sys.modules.pop("main", None)  # сбросить, если успел подтянуться stale-модуль
    try:
        from main import _cli_entry
    except ModuleNotFoundError as e:
        print(f"sea: не удалось загрузить агента ({e}).\n"
              f"Запускай из каталога проекта (где лежат config.yml и .env), либо переустанови: "
              f"uv sync.", file=sys.stderr)
        raise SystemExit(2)
    _cli_entry()


if __name__ == "__main__":
    main_cli()
