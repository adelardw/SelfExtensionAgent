"""
CLI-настройки: `config.local.yml` МЕРДЖИТСЯ поверх `config.yml` (запрос юзера: «CLI должна
иметь конфиг отдельный либо метчиться с config.yaml; изменения в CLI вписываются в конфиг»).

Почему отдельный файл, а не запись в config.yml: OmegaConf.save теряет комментарии, а
config.yml — комментированный источник правды. Поэтому config.yml только читается;
всё, что юзер меняет ИЗ CLI (/model и т.п.), пишется в config.local.yml (в .gitignore)
и при следующем старте автоматически применяется через merge.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from . import config_paths

# BASE — cwd config.yml ИЛИ пакетный дефолт (резолвер): установленный `sea` (uv tool / uv pip)
# работает в ЛЮБОМ каталоге, а не только в репо. Базовые модели — в пакетном config.yml.
BASE = config_paths.base_config_path()

# LOCAL (CLI-настройки: api_key, провайдер, base_url, гранты): cwd `config.local.yml` если есть
# (проектный override, как раньше) ИНАЧЕ ГЛОБАЛЬНЫЙ ~/.config/sea/config.local.yml. Поэтому в новом
# проекте `sea key`/`/key` пишет в глобальный → ключ работает во ВСЕХ проектах. Модульные BASE/LOCAL
# монкейпатчатся в тестах — load_merged/set_cli ходят именно через них.
_proj = config_paths.project_local_path()
LOCAL = _proj if _proj.exists() else config_paths.global_local_path()

# config.local.yml хранит api_key и HITL-гранты. Запись делаем атомарно (temp+fsync+os.replace)
# под локом: иначе конкурентный set_cli (CLI + фоновый персист гранта) мог оставить полу-записанный
# YAML, обнуляющий ключ/гранты при следующем merge (баг ревью CON-2 — единственный конфиг-писатель,
# не приведённый к atomic+lock; intent/prompt_store/registry уже приведены).
_SAVE_LOCK = threading.Lock()


def _atomic_save(cfg, path: Path) -> None:
    """Атомарная запись YAML (temp+fsync+os.replace). Лок держит ВЫЗЫВАЮЩИЙ (RMW — одна критсекция);
    сам не захватывает _SAVE_LOCK, чтобы не было дедлока с нереентрантным Lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # атомарная подмена — читатель видит либо старый, либо новый файл
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_merged():
    """base (cwd config.yml | пакетный дефолт) + LOCAL поверх (cwd-проектный или глобальный). Поздний
    выигрывает. Битый local не ломает запуск."""
    base = OmegaConf.load(BASE) if Path(BASE).exists() else OmegaConf.create({})
    if Path(LOCAL).exists():
        try:
            return OmegaConf.merge(base, OmegaConf.load(LOCAL))
        except Exception:  # noqa: BLE001 — битый local не должен ломать запуск
            return base
    return base


def get_cli(key: str, default: Any = None) -> Any:
    """Чтение CLI-настройки (раздел cli.*) из смерженного конфига."""
    return OmegaConf.select(load_merged(), f"cli.{key}", default=default)


def set_cli(key: str, value: Any) -> None:
    """Персист изменения из CLI в LOCAL (cli.<key>): cwd config.local.yml в проекте, иначе глобальный
    ~/.config/sea/config.local.yml (ключ/провайдер — на все проекты). Атомарно + под локом (RMW)."""
    path = Path(LOCAL)
    if path == config_paths.global_local_path():
        config_paths.ensure_user_dir()  # каталог глобального конфига → 0700 (в нём api_key)
    with _SAVE_LOCK:
        local = OmegaConf.load(path) if path.exists() else OmegaConf.create({})
        OmegaConf.update(local, f"cli.{key}", value, merge=True)
        _atomic_save(local, path)
