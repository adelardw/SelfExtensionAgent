"""
Резолвер путей конфига — чтобы `sea`, установленный ПАКЕТОМ (глобально через `uv tool`/`pipx` или
в venv через `uv pip install`), работал в ЛЮБОМ каталоге, а не только в исходном репозитории.

Три источника, мердж по приоритету (поздний выигрывает):
  1. БАЗА (`config.yml`)         — дефолтные модели/настройки. cwd-файл (project-rooted override),
                                   иначе ПАКЕТНЫЙ дефолт (ships с пакетом). «Базовые модели остаются».
  2. ГЛОБАЛЬНЫЙ local            — `~/.config/sea/config.local.yml`: api_key, провайдер, base_url —
                                   задаются ИЗ CLI один раз и работают во ВСЕХ проектах.
  3. ПРОЕКТНЫЙ local             — `./config.local.yml` в cwd (per-project override, как раньше).

Переопределения: env `SEA_CONFIG_DIR` (каталог глобального конфига), `XDG_CONFIG_HOME`.
"""
from __future__ import annotations

import os
from pathlib import Path

# Пакетный дефолт config.yml. config_paths.py лежит в <root>/src/ → <root>/config.yml. Для editable —
# это корень репо; для wheel — корень site-packages (config.yml кладётся туда через force-include в
# pyproject). Так один и тот же путь валиден и для editable, и для установленного пакета.
_PKG_DEFAULT = Path(__file__).resolve().parent.parent / "config.yml"


def base_config_path() -> Path:
    """config.yml: cwd (project-rooted override) → пакетный дефолт. Никогда не падает на отсутствии
    cwd-файла в чужом проекте."""
    cwd = Path("config.yml")
    return cwd if cwd.exists() else _PKG_DEFAULT


def user_config_dir() -> Path:
    """Каталог глобального пользовательского конфига: SEA_CONFIG_DIR → XDG_CONFIG_HOME/sea → ~/.config/sea."""
    env = os.getenv("SEA_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.getenv("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else (Path.home() / ".config")
    return base / "sea"


def global_local_path() -> Path:
    """Глобальный config.local.yml (api_key/провайдер/base_url — на всех проектах)."""
    return user_config_dir() / "config.local.yml"


def ensure_user_dir() -> Path:
    """Создать каталог глобального конфига с правами 0700 (в нём лежит api_key). Файл и так пишется
    0600 (mkstemp), но 0700 на каталоге — defense-in-depth на мультиюзер-машине. Зовётся при ЗАПИСИ."""
    d = user_config_dir()
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def project_local_path() -> Path:
    """Проектный config.local.yml (cwd) — per-project override."""
    return Path("config.local.yml")
