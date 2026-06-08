"""
Авто-обновление зависимостей с проверкой и откатом.

Логика безопасного апдейта (никогда не оставляем окружение сломанным):
  1. снимок uv.lock;
  2. `uv lock --upgrade` + `uv sync`;
  3. health-check: все навыки из реестра обязаны импортироваться;
  4. если что-то сломалось — восстановить uv.lock и `uv sync` (rollback).

Запуск разовый: `python -m src.maintenance`.
Для периодичности используйте внешний планировщик (cron / Claude /schedule).
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from ..tools.skill_creation import SKILLS_DIR, _load_registry

LOCK = Path("uv.lock")
LOCK_BAK = Path("uv.lock.bak")


def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return p.returncode == 0, (p.stdout + p.stderr)[-2000:]
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _skills_healthy() -> tuple[bool, list[str]]:
    """Проверяет, что каждый навык с кодом импортируется без ошибок."""
    broken = []
    for name, meta in _load_registry().items():
        if not meta.get("has_tools"):
            continue
        py = SKILLS_DIR / name / f"{name}.py"
        if not py.exists():
            continue
        try:
            mod_name = f"_health.{name}"
            spec = importlib.util.spec_from_file_location(mod_name, str(py))
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as e:  # noqa: BLE001
            broken.append(f"{name}: {type(e).__name__}: {e}")
        finally:
            sys.modules.pop(f"_health.{name}", None)
    return (len(broken) == 0), broken


def run_update(dry_run: bool = False) -> dict:
    if not LOCK.exists():
        return {"status": "error", "reason": "uv.lock не найден"}

    # baseline health — чтобы не списать на апдейт ранее сломанные навыки
    base_ok, base_broken = _skills_healthy()

    if dry_run:
        ok, out = _run(["uv", "lock", "--upgrade", "--dry-run"])
        return {"status": "dry_run", "ok": ok, "output": out, "baseline_broken": base_broken}

    shutil.copy(LOCK, LOCK_BAK)
    ok, out = _run(["uv", "lock", "--upgrade"])
    if ok:
        ok, out = _run(["uv", "sync"])
    if not ok:
        shutil.copy(LOCK_BAK, LOCK)
        _run(["uv", "sync"])
        LOCK_BAK.unlink(missing_ok=True)
        return {"status": "rollback", "reason": "uv lock/sync упал", "output": out}

    healthy, broken = _skills_healthy()
    new_breaks = [b for b in broken if b not in base_broken]
    if new_breaks:
        shutil.copy(LOCK_BAK, LOCK)
        _run(["uv", "sync"])
        LOCK_BAK.unlink(missing_ok=True)
        return {"status": "rollback", "reason": "апдейт сломал навыки", "broken": new_breaks}

    LOCK_BAK.unlink(missing_ok=True)
    return {"status": "updated", "skills_ok": healthy, "preexisting_broken": base_broken}
