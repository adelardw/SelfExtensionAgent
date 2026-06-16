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

BASE = Path("config.yml")
LOCAL = Path("config.local.yml")

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
    """Персист изменения из CLI: пишется ТОЛЬКО в config.local.yml (cli.<key>). Атомарно+под локом."""
    with _SAVE_LOCK:  # вся read-modify-write — одна критсекция (lost-update недопустим для api_key/грантов)
        local = OmegaConf.load(LOCAL) if LOCAL.exists() else OmegaConf.create({})
        OmegaConf.update(local, f"cli.{key}", value, merge=True)
        _atomic_save(local, LOCAL)
