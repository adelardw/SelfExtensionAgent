"""
CLI-настройки: `config.local.yml` МЕРДЖИТСЯ поверх `config.yml` (запрос юзера: «CLI должна
иметь конфиг отдельный либо метчиться с config.yaml; изменения в CLI вписываются в конфиг»).

Почему отдельный файл, а не запись в config.yml: OmegaConf.save теряет комментарии, а
config.yml — комментированный источник правды. Поэтому config.yml только читается;
всё, что юзер меняет ИЗ CLI (/model и т.п.), пишется в config.local.yml (в .gitignore)
и при следующем старте автоматически применяется через merge.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

BASE = Path("config.yml")
LOCAL = Path("config.local.yml")


def load_merged():
    """config.yml + config.local.yml поверх (local выигрывает)."""
    base = OmegaConf.load(BASE) if BASE.exists() else OmegaConf.create({})
    if LOCAL.exists():
        try:
            return OmegaConf.merge(base, OmegaConf.load(LOCAL))
        except Exception:  # noqa: BLE001 — битый local не должен ломать запуск
            return base
    return base


def get_cli(key: str, default: Any = None) -> Any:
    """Чтение CLI-настройки (раздел cli.*) из смерженного конфига."""
    return OmegaConf.select(load_merged(), f"cli.{key}", default=default)


def set_cli(key: str, value: Any) -> None:
    """Персист изменения из CLI: пишется ТОЛЬКО в config.local.yml (cli.<key>)."""
    local = OmegaConf.load(LOCAL) if LOCAL.exists() else OmegaConf.create({})
    OmegaConf.update(local, f"cli.{key}", value, merge=True)
    OmegaConf.save(local, LOCAL)
