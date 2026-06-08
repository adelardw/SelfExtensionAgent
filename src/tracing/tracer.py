"""
Полный трейсинг агента по нодам графа.

Зачем: implicit-loop и само-улучшение должны опираться на ФАКТЫ — какая нода
вызывалась, сколько шла, упала ли. Без этого «агент находит свои косяки» —
пустые слова. Пишем компактные спаны в ОТДЕЛЬНУЮ SQLite (data/traces.db),
чтобы трейсы можно было ротировать/чистить независимо от долгой памяти.

Критик-замечание: полноценный observability-сервис (LangSmith/OTel) — это уже
инфраструктура. На текущем масштабе локальный SQLite-трейсер дешевле, приватнее
(данные не уходят наружу — важно для on-device цели) и достаточен. Апгрейд до
LangSmith — одна обёртка-callback, когда реально понадобится распределённость.
"""
from __future__ import annotations

import contextvars
import functools
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

# run_id текущего прохода графа — живёт в пределах одной async-цепочки invoke.
_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")


def new_run() -> str:
    rid = uuid.uuid4().hex[:12]
    _run_id.set(rid)
    return rid


def current_run() -> str:
    return _run_id.get() or "unknown"


class TraceStore:
    def __init__(self, db_path: str = "data/traces.db", keep_spans: int = 20000):
        self.keep_spans = keep_spans
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS spans (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id    TEXT,
                node      TEXT NOT NULL,
                ts        REAL NOT NULL,
                duration_ms REAL NOT NULL,
                status    TEXT NOT NULL,
                error     TEXT DEFAULT '',
                out_keys  TEXT DEFAULT '',
                output    TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_spans_node ON spans(node);
            CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id);
            """
        )
        try:  # миграция для существующих БД
            self._conn.execute("ALTER TABLE spans ADD COLUMN output TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        self._conn.commit()

    def record(self, run_id: str, node: str, duration_ms: float, status: str,
               error: str = "", out_keys: str = "", output: str = "") -> None:
        self._conn.execute(
            "INSERT INTO spans (run_id, node, ts, duration_ms, status, error, out_keys, output) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, node, time.time(), duration_ms, status, error[:500], out_keys[:300], output[:300]),
        )
        self._conn.commit()

    def run_trace(self, run_id: str) -> list[tuple[str, str]]:
        """Упорядоченный forward-проход прогона: [(node, output_snapshot)] — основа edge-gradient."""
        rows = self._conn.execute(
            "SELECT node, output FROM spans WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [(r["node"], r["output"] or "") for r in rows]

    def prune(self) -> int:
        """Ротация: оставляем последние keep_spans строк. Возвращает число удалённых."""
        row = self._conn.execute("SELECT COUNT(*) AS c FROM spans").fetchone()
        total = int(row["c"]) if row else 0
        if total <= self.keep_spans:
            return 0
        cutoff = self._conn.execute(
            "SELECT id FROM spans ORDER BY id DESC LIMIT 1 OFFSET ?", (self.keep_spans,)
        ).fetchone()
        if not cutoff:
            return 0
        self._conn.execute("DELETE FROM spans WHERE id <= ?", (cutoff["id"],))
        self._conn.commit()
        return total - self.keep_spans

    # ── аналитика для самодиагностики ──
    def node_stats(self, since_hours: float = 24.0) -> list[sqlite3.Row]:
        since = time.time() - since_hours * 3600
        return self._conn.execute(
            "SELECT node, COUNT(*) AS calls, AVG(duration_ms) AS avg_ms, MAX(duration_ms) AS max_ms, "
            "SUM(status!='ok') AS errors FROM spans WHERE ts>=? GROUP BY node ORDER BY avg_ms DESC",
            (since,),
        ).fetchall()

    def retry_storms(self, since_hours: float = 24.0, threshold: int = 4) -> list[sqlite3.Row]:
        """Проходы, где один шаг крутился слишком много раз (признак залипания)."""
        since = time.time() - since_hours * 3600
        return self._conn.execute(
            "SELECT run_id, COUNT(*) AS step_calls FROM spans "
            "WHERE ts>=? AND node='step_executor' GROUP BY run_id HAVING step_calls>=? ORDER BY step_calls DESC",
            (since, threshold),
        ).fetchall()

    def nodes_for_runs(self, run_ids: list[str]) -> dict[str, list[str]]:
        """Карта run_id → список активированных нод (для backward credit assignment)."""
        if not run_ids:
            return {}
        ph = ",".join("?" * len(run_ids))
        rows = self._conn.execute(
            f"SELECT run_id, node FROM spans WHERE run_id IN ({ph}) ORDER BY id", tuple(run_ids)
        ).fetchall()
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["run_id"], []).append(r["node"])
        return out

    def close(self) -> None:
        self._conn.close()


# Глобальный трейс-стор процесса.
trace_store = TraceStore()


def traced(node_name: str, fn: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
    """Оборачивает async-ноду графа: замеряет время, статус и пишет спан."""

    @functools.wraps(fn)
    async def wrapper(state):
        rid = current_run()
        t0 = time.perf_counter()
        status, error, out = "ok", "", {}
        try:
            out = await fn(state)
            return out
        except Exception as e:  # noqa: BLE001
            status, error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            dur = (time.perf_counter() - t0) * 1000
            try:
                d = out or {}
                # компактный снимок выхода ноды (без огромных полей вроде memory_context)
                snap = "; ".join(
                    f"{k}={str(v)[:70]}" for k, v in d.items()
                    if k not in ("memory_context", "messages", "chat_history", "skill_prompts")
                )[:300]
                trace_store.record(rid, node_name, dur, status, error, ",".join(d.keys()), snap)
            except Exception:  # noqa: BLE001
                pass

    return wrapper
