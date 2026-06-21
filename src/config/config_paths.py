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


def bootstrap_frozen() -> None:
    """Для УПАКОВАННОГО приложения (.app/.exe, PyInstaller). При запуске из Launchpad/Dock/Finder
    cwd = '/' (read-only), а код пишет ОТНОСИТЕЛЬНЫЕ data/, traces.db, config.local.yml → краш
    (OSError: Read-only file system: 'data'). Переходим в writable per-user рабочую папку и кладём
    туда дефолтный config.yml из бандла. Зовётся entrypoint'ом (main.py/desktop.py) ДО тяжёлых
    импортов (они читают config.yml/создают data/). Из исходников (не frozen) — no-op."""
    import shutil
    import sys
    if not getattr(sys, "frozen", False):
        return
    if sys.platform == "darwin":
        workdir = Path.home() / "Library" / "Application Support" / "SEA"
    elif os.name == "nt":
        workdir = Path(os.getenv("APPDATA", str(Path.home()))) / "SEA"
    else:
        workdir = Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "SEA"
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)  # теперь относительные data/, config.local.yml резолвятся в writable папку
    # PATH в .app из Launchpad УРЕЗАН (/usr/bin:/bin:…) — нет /opt/homebrew/bin → subprocess не
    # находит brew-инструменты (ffmpeg для голоса/медиа и пр.). Дополняем стандартными bin-путями.
    _extra = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin",
              str(Path.home() / "homebrew" / "bin")]
    cur = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(dict.fromkeys([*cur, *_extra]))  # дедуп, порядок сохранён
    # liteparse грузит libpdfium.dylib через dlopen и ищет её в PDFIUM_LIB_PATH. В бандле она лежит
    # в _MEIPASS/liteparse → указываем туда (иначе парсинг PDF падает PanicException в .app).
    bundle = Path(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    if not os.environ.get("PDFIUM_LIB_PATH"):
        for cand in (bundle / "liteparse", bundle):
            if (cand / "libpdfium.dylib").exists():
                os.environ["PDFIUM_LIB_PATH"] = str(cand)
                break
    bundle = Path(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    if not (workdir / "config.yml").exists() and (bundle / "config.yml").exists():
        try:
            shutil.copy(bundle / "config.yml", workdir / "config.yml")
        except OSError:
            pass
