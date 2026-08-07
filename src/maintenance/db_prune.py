"""Прунинг checkpoints.db (найдено `sea doctor`: 258МБ на 531 тред).

LangGraph-чекпоинтер пишет чекпоинт на КАЖДЫЙ супер-шаг каждого треда (~14/тред) — для
возобновления треда нужен только ПОСЛЕДНИЙ (checkpoint_id у LangGraph монотонный → MAX =
самый свежий). Держим последний чекпоинт на (thread_id, ns) + его writes, остальное режем,
затем VACUUM. Перед правкой — бэкап-копия рядом (.bak): операция необратимая, но потери
ограничены ИСТОРИЕЙ шагов (не текущим состоянием тредов; сами чаты живут в chat_store).
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

_KEEP_SUBQ = ("SELECT thread_id, checkpoint_ns, MAX(checkpoint_id) "
              "FROM checkpoints GROUP BY thread_id, checkpoint_ns")


def prune_checkpoints(db_path: str = "data/checkpoints.db", backup: bool = True) -> dict:
    """Оставить последний чекпоинт каждого треда. Возвращает сводку с размерами до/после."""
    p = Path(db_path)
    if not p.exists():
        return {"status": "нет файла", "path": str(p)}
    before_mb = p.stat().st_size / 1024 / 1024
    if backup:
        shutil.copy2(p, p.with_suffix(".db.bak"))
    conn = sqlite3.connect(p)
    try:
        threads = conn.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints").fetchone()[0]
        dropped_cp = conn.execute(
            f"DELETE FROM checkpoints WHERE (thread_id, checkpoint_ns, checkpoint_id) "
            f"NOT IN ({_KEEP_SUBQ})").rowcount
        dropped_wr = conn.execute(
            f"DELETE FROM writes WHERE (thread_id, checkpoint_ns, checkpoint_id) "
            f"NOT IN ({_KEEP_SUBQ})").rowcount
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    after_mb = p.stat().st_size / 1024 / 1024
    return {"status": "ok", "threads": threads, "dropped_checkpoints": dropped_cp,
            "dropped_writes": dropped_wr, "before_mb": round(before_mb, 1),
            "after_mb": round(after_mb, 1),
            "backup": str(p.with_suffix(".db.bak")) if backup else ""}
