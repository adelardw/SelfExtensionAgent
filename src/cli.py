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
  sea                       запустить полноэкранный TUI (диалог в терминале)
  sea "<задача>"            выполнить одну задачу и выйти (one-shot)
  sea --auto "<задача>"     one-shot без подтверждений (автономно)
  sea init                  создать .sea/ + стартовые SEA.md/MEMORY.md/MCP.md

Настройка (пишется в ~/.config/sea/config.local.yml — работает во ВСЕХ проектах):
  sea key <API_KEY>         задать ключ провайдера
  sea provider openrouter|ollama [base_url]   выбрать провайдера
  sea config                показать текущую конфигурацию (провайдер, ключ, пути)

  sea --version, -V         версия      ·      sea --help, -h      справка

Установлен ПАКЕТОМ (uv tool / uv pip) — работает в любом каталоге: базовые модели берутся из
пакетного config.yml; cwd-config.yml (если есть) переопределяет его. Ключ: env (.env) → `sea key`.
В TUI те же настройки: /key <KEY> · /provider <name> [base_url] · /model · /config."""


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
    # Настройка ИЗ CLI (без импорта агента) — пишется в ГЛОБАЛЬНЫЙ ~/.config/sea/config.local.yml,
    # поэтому работает во всех проектах. Базовые модели остаются из пакетного config.yml.
    if first in ("key", "login"):
        from .cli_config import set_cli
        from . import config_paths
        # Ключ как АРГУМЕНТ виден в `ps aux` и истории шелла (~/.zsh_history). Безопаснее — без
        # аргумента: спросим скрытым вводом (getpass, без эха, не в истории). `-` тоже → stdin.
        if len(args) >= 2 and args[1] != "-":
            print("⚠ ключ передан аргументом — он попадёт в историю шелла и `ps`. Безопаснее: "
                  "`sea key` (скрытый ввод) или env OPEN_ROUTER_API_KEY.")
            key = args[1].strip()
        else:
            import getpass
            key = getpass.getpass("API key (ввод скрыт): ").strip()
        if not key:
            print("Пусто — ключ не сохранён.")
            raise SystemExit(2)
        set_cli("api_key", key)
        print(f"✓ API-ключ сохранён → {config_paths.global_local_path()} (0600)")
        return
    if first == "provider":
        from .cli_config import set_cli
        from . import config_paths
        if len(args) < 2 or args[1] not in ("openrouter", "ollama"):
            print("Использование: sea provider openrouter|ollama [base_url]")
            raise SystemExit(2)
        set_cli("provider", args[1])
        if len(args) > 2:
            set_cli("base_url", args[2].strip())
        print(f"✓ Провайдер: {args[1]}" + (f" · base_url: {args[2]}" if len(args) > 2 else "")
              + f"  → {config_paths.global_local_path()}")
        return
    if first == "config":
        from .cli_config import get_cli
        from . import config_paths
        print("sea config:")
        print(f"  base config : {config_paths.base_config_path()}")
        print(f"  user config : {config_paths.global_local_path()}")
        print(f"  provider    : {get_cli('provider') or 'openrouter (default)'}")
        print(f"  base_url    : {get_cli('base_url') or '(default)'}")
        has_env = bool(__import__('os').getenv('OPEN_ROUTER_API_KEY') or __import__('os').getenv('OPENAI_API_KEY'))
        print(f"  api key     : {'env (.env)' if has_env else ('set (user config)' if get_cli('api_key') else 'NOT SET — run: sea key <KEY>')}")
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
