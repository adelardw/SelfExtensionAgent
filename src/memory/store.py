"""
MemoryStore — постоянная память агента поверх SQLite.

Три вида памяти (по таксономии "Memory in the Age of AI Agents", 2025):
  • episodes    — эпизодическая: что спрашивали, что ответили, чем кончилось.
                  Одновременно это trajectory-store для будущего fine-tuning.
  • facts       — семантическая / персонализация: устойчивые факты о пользователе.
  • reflections — выводы высокого порядка, синтезированные из эпизодов
                  (механика memory-stream из Generative Agents).

Извлечение (recall) скорит воспоминания по взвешенной сумме
recency + relevance + importance — ровно как в Generative Agents.

Хранилище намеренно простое: один пользователь, локальный объём данных малы,
поэтому relevance считается линейным проходом (token-overlap или косинус по
эмбеддингам). Никаких внешних векторных БД на этом этапе — graceful by design.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .embedder import Embedder, NullEmbedder, cosine
from .vector_index import VectorIndex, turbovec_available

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Период полураспада recency (сек). ~3 дня: свежее помнится лучше.
_RECENCY_HALFLIFE = 3 * 24 * 3600.0

# Веса скоринга recall (Generative Agents).
_W_RECENCY = 1.0
_W_RELEVANCE = 1.5
_W_IMPORTANCE = 1.0


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _overlap(a: str, b: str) -> float:
    """Jaccard-похожесть по токенам — keyword-релевантность без эмбеддингов."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


class MemoryStore:
    """Потокобезопасный (check_same_thread=False) синхронный стор для одного процесса."""

    def __init__(self, db_path: str, embedder: Optional[Embedder] = None):
        self.embedder = embedder or NullEmbedder()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        # TurboVec ANN-индекс по эпизодам — строится лениво при первом векторе.
        self._vindex: Optional[VectorIndex] = None
        self._vindex_ready = not (self.embedder.enabled and turbovec_available())

    # ── schema ────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT NOT NULL,
                ts        REAL NOT NULL,
                query     TEXT NOT NULL,
                answer    TEXT NOT NULL,
                route     TEXT,
                skills    TEXT,
                confidence REAL,
                outcome   TEXT,
                embedding TEXT
            );
            CREATE TABLE IF NOT EXISTS facts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT NOT NULL,
                key       TEXT NOT NULL,
                value     TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                ts        REAL NOT NULL,
                source_episode INTEGER,
                embedding TEXT
            );
            CREATE TABLE IF NOT EXISTS reflections (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT NOT NULL,
                insight   TEXT NOT NULL,
                importance REAL DEFAULT 0.6,
                ts        REAL NOT NULL,
                embedding TEXT
            );
            CREATE TABLE IF NOT EXISTS goals (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT NOT NULL,
                aim       TEXT NOT NULL,
                status    TEXT NOT NULL DEFAULT 'active',
                ts        REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS summaries (
                user_id   TEXT PRIMARY KEY,
                text      TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_edges (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT NOT NULL,
                src_type  TEXT NOT NULL,
                src_id    INTEGER NOT NULL,
                dst_type  TEXT NOT NULL,
                dst_id    INTEGER NOT NULL,
                relation  TEXT NOT NULL DEFAULT 'related',
                ts        REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON memory_edges(user_id, src_type, src_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id, ts);
            CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
            CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id, status);
            """
        )
        # Лёгкие миграции (ADD COLUMN падает, если колонка уже есть — игнорируем).
        for ddl in (
            "ALTER TABLE episodes ADD COLUMN feedback TEXT DEFAULT ''",
            "ALTER TABLE goals ADD COLUMN criteria TEXT DEFAULT '[]'",
            "ALTER TABLE facts ADD COLUMN tags TEXT DEFAULT '[]'",
            "ALTER TABLE episodes ADD COLUMN tags TEXT DEFAULT '[]'",
            "ALTER TABLE episodes ADD COLUMN run_id TEXT DEFAULT ''",
            "ALTER TABLE episodes ADD COLUMN mode TEXT DEFAULT ''",
        ):
            try:
                self._conn.execute(ddl)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    # ── writes ────────────────────────────────────────────────────────

    def _emb_json(self, text: str) -> Optional[str]:
        vec = self.embedder.embed(text)
        return json.dumps(vec) if vec else None

    def _ensure_vindex(self, dim: int) -> None:
        """Лениво создаёт ANN-индекс и заполняет его из уже сохранённых эпизодов."""
        if self._vindex_ready:
            return
        try:
            self._vindex = VectorIndex(dim)
            for ep in self._conn.execute(
                "SELECT id, embedding FROM episodes WHERE embedding IS NOT NULL"
            ).fetchall():
                try:
                    self._vindex.add(ep["id"], json.loads(ep["embedding"]))
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            print(f"[MemoryStore] vindex init failed: {e}")
            self._vindex = None
        self._vindex_ready = True

    def add_episode(
        self,
        user_id: str,
        query: str,
        answer: str,
        route: str = "",
        skills: Optional[list[str]] = None,
        confidence: float = 0.0,
        outcome: str = "ok",
        feedback: str = "",
        run_id: str = "",
        mode: str = "",
    ) -> int:
        vec = self.embedder.embed(query) if self.embedder.enabled else None
        emb_json = json.dumps(vec) if vec else None
        cur = self._conn.execute(
            "INSERT INTO episodes (user_id, ts, query, answer, route, skills, confidence, outcome, embedding, feedback, run_id, mode) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                user_id,
                time.time(),
                query,
                answer,
                route,
                json.dumps(skills or [], ensure_ascii=False),
                confidence,
                outcome,
                emb_json,
                feedback,
                run_id,
                mode,
            ),
        )
        self._conn.commit()
        ep_id = cur.lastrowid

        if vec:
            self._ensure_vindex(len(vec))
            if self._vindex:
                self._vindex.add(ep_id, vec)
        return ep_id

    def add_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        importance: float = 0.5,
        source_episode: Optional[int] = None,
        tags: Optional[list[str]] = None,
    ) -> int:
        """Upsert по (user_id, key): новый факт перезаписывает старое. Возвращает id факта."""
        existing = self._conn.execute(
            "SELECT id FROM facts WHERE user_id=? AND lower(key)=lower(?)",
            (user_id, key),
        ).fetchone()
        emb = self._emb_json(f"{key}: {value}")
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        if existing:
            self._conn.execute(
                "UPDATE facts SET value=?, importance=?, ts=?, source_episode=?, embedding=?, tags=? WHERE id=?",
                (value, importance, time.time(), source_episode, emb, tags_json, existing["id"]),
            )
            fact_id = existing["id"]
        else:
            cur = self._conn.execute(
                "INSERT INTO facts (user_id, key, value, importance, ts, source_episode, embedding, tags) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (user_id, key, value, importance, time.time(), source_episode, emb, tags_json),
            )
            fact_id = cur.lastrowid
        self._conn.commit()
        return fact_id

    # ── graph edges (взаимосвязанная память) ─────────────────────────

    def add_edge(self, user_id: str, src_type: str, src_id: int, dst_type: str, dst_id: int, relation: str = "related") -> None:
        self._conn.execute(
            "INSERT INTO memory_edges (user_id, src_type, src_id, dst_type, dst_id, relation, ts) VALUES (?,?,?,?,?,?,?)",
            (user_id, src_type, src_id, dst_type, dst_id, relation, time.time()),
        )
        self._conn.commit()

    def neighbors(self, user_id: str, node_type: str, node_id: int) -> list[sqlite3.Row]:
        """1-hop соседи узла в обе стороны (для связного recall / multi-hop)."""
        return self._conn.execute(
            "SELECT dst_type AS type, dst_id AS id, relation FROM memory_edges "
            "WHERE user_id=? AND src_type=? AND src_id=? "
            "UNION SELECT src_type AS type, src_id AS id, relation FROM memory_edges "
            "WHERE user_id=? AND dst_type=? AND dst_id=?",
            (user_id, node_type, node_id, user_id, node_type, node_id),
        ).fetchall()

    # ── standing goals (persistent task context) ─────────────────────

    def get_active_goal(self, user_id: str):
        return self._conn.execute(
            "SELECT * FROM goals WHERE user_id=? AND status='active' ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()

    def set_goal(self, user_id: str, aim: str, criteria: Optional[list[str]] = None) -> None:
        """Одна активная «стоящая» цель + её rubric (критерии успеха): upsert активной."""
        active = self.get_active_goal(user_id)
        now = time.time()
        crit_json = json.dumps(criteria or [], ensure_ascii=False)
        if active:
            # Критерии обновляем только если переданы непустые — иначе сохраняем прежние.
            if criteria:
                self._conn.execute(
                    "UPDATE goals SET aim=?, criteria=?, updated_at=? WHERE id=?",
                    (aim, crit_json, now, active["id"]),
                )
            else:
                self._conn.execute(
                    "UPDATE goals SET aim=?, updated_at=? WHERE id=?", (aim, now, active["id"])
                )
        else:
            self._conn.execute(
                "INSERT INTO goals (user_id, aim, status, ts, updated_at, criteria) VALUES (?,?,'active',?,?,?)",
                (user_id, aim, now, now, crit_json),
            )
        self._conn.commit()

    @staticmethod
    def goal_criteria(goal_row) -> list[str]:
        """Безопасно парсит rubric-критерии из строки goals."""
        if not goal_row or "criteria" not in goal_row.keys():
            return []
        try:
            return json.loads(goal_row["criteria"] or "[]")
        except Exception:  # noqa: BLE001
            return []

    def close_active_goal(self, user_id: str) -> None:
        self._conn.execute(
            "UPDATE goals SET status='done', updated_at=? WHERE user_id=? AND status='active'",
            (time.time(), user_id),
        )
        self._conn.commit()

    def add_reflection(self, user_id: str, insight: str, importance: float = 0.6) -> None:
        self._conn.execute(
            "INSERT INTO reflections (user_id, insight, importance, ts, embedding) VALUES (?,?,?,?,?)",
            (user_id, insight, importance, time.time(), self._emb_json(insight)),
        )
        self._conn.commit()

    # ── reads ─────────────────────────────────────────────────────────

    def get_facts(self, user_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM facts WHERE user_id=? ORDER BY importance DESC, ts DESC",
            (user_id,),
        ).fetchall()

    def recent_episodes(self, user_id: str, n: int = 5) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM episodes WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, n),
        ).fetchall()

    def get_failures(self, n: int = 20, user_id: Optional[str] = None) -> list[sqlite3.Row]:
        """Слабые эпизоды (outcome != 'ok') для self-learning — по всем юзерам или одному."""
        if user_id:
            return self._conn.execute(
                "SELECT * FROM episodes WHERE outcome!='ok' AND user_id=? ORDER BY ts DESC LIMIT ?",
                (user_id, n),
            ).fetchall()
        return self._conn.execute(
            "SELECT * FROM episodes WHERE outcome!='ok' ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()

    def get_successes(self, n: int = 40, user_id: Optional[str] = None) -> list[sqlite3.Row]:
        """Валидированные успехи (для дифференциального credit assignment)."""
        if user_id:
            return self._conn.execute(
                "SELECT * FROM episodes WHERE outcome='ok' AND confidence>0 AND user_id=? ORDER BY ts DESC LIMIT ?",
                (user_id, n),
            ).fetchall()
        return self._conn.execute(
            "SELECT * FROM episodes WHERE outcome='ok' AND confidence>0 ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()

    def failure_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM episodes WHERE outcome!='ok'"
        ).fetchone()
        return int(row["c"]) if row else 0

    # ── local context: rolling summary (SummaryCtx) ──────────────────

    def get_summary(self, user_id: str) -> str:
        row = self._conn.execute(
            "SELECT text FROM summaries WHERE user_id=?", (user_id,)
        ).fetchone()
        return row["text"] if row else ""

    def set_summary(self, user_id: str, text: str) -> None:
        self._conn.execute(
            "INSERT INTO summaries (user_id, text, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET text=excluded.text, updated_at=excluded.updated_at",
            (user_id, text, time.time()),
        )
        self._conn.commit()

    # ── quality degradation tracking ─────────────────────────────────

    def quality_trend(self, user_id: str, window: int = 10) -> dict:
        """
        Сравнивает среднюю уверенность недавней половины валидированных эпизодов
        с предыдущей. Возвращает {trend, recent_avg, prev_avg, n}.
        Учитываются только эпизоды с реальной валидацией (confidence > 0).
        """
        rows = self._conn.execute(
            "SELECT confidence FROM episodes WHERE user_id=? AND confidence>0 "
            "ORDER BY ts DESC LIMIT ?",
            (user_id, window),
        ).fetchall()
        confs = [r["confidence"] for r in rows]
        if len(confs) < 4:
            return {"trend": "unknown", "recent_avg": None, "prev_avg": None, "n": len(confs)}
        half = len(confs) // 2
        recent = confs[:half]          # самые свежие (DESC)
        prev = confs[half:]
        ra, pa = sum(recent) / len(recent), sum(prev) / len(prev)
        if ra < pa - 0.07:
            trend = "declining"
        elif ra > pa + 0.07:
            trend = "improving"
        else:
            trend = "stable"
        return {"trend": trend, "recent_avg": round(ra, 3), "prev_avg": round(pa, 3), "n": len(confs)}

    def episode_count(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM episodes WHERE user_id=?", (user_id,)
        ).fetchone()
        return int(row["c"]) if row else 0

    # ── retrieval scoring (Generative Agents) ─────────────────────────

    def _relevance(self, query: str, text: str, emb_json: Optional[str]) -> float:
        if self.embedder.enabled and emb_json:
            qvec = self.embedder.embed(query)
            if qvec:
                try:
                    return cosine(qvec, json.loads(emb_json))
                except Exception:  # noqa: BLE001
                    pass
        return _overlap(query, text)

    @staticmethod
    def _recency(ts: float) -> float:
        age = max(0.0, time.time() - ts)
        return math.exp(-age / _RECENCY_HALFLIFE)

    def _score(self, query: str, text: str, ts: float, importance: float, emb_json: Optional[str]) -> float:
        return (
            _W_RECENCY * self._recency(ts)
            + _W_RELEVANCE * self._relevance(query, text, emb_json)
            + _W_IMPORTANCE * importance
        )

    def _rank_episodes(self, user_id: str, query: str, k: int) -> list[sqlite3.Row]:
        """
        Гибридное ранжирование эпизодов:
          • если активен TurboVec-индекс (эмбеддинги + достаточно данных) —
            ANN-кандидаты по вектору, затем дореранк по recency+importance;
          • иначе — линейный скоринг recency+relevance(token)+importance.
        """
        if self._vindex and self._vindex.active and self.embedder.enabled:
            qvec = self.embedder.embed(query)
            if qvec:
                user_ids = [
                    r["id"] for r in self._conn.execute(
                        "SELECT id FROM episodes WHERE user_id=?", (user_id,)
                    ).fetchall()
                ]
                hits = self._vindex.search(qvec, k * 3, allowed=user_ids)
                if hits:
                    id2rel = dict(hits)
                    placeholders = ",".join("?" * len(id2rel))
                    rows = self._conn.execute(
                        f"SELECT * FROM episodes WHERE id IN ({placeholders})", tuple(id2rel)
                    ).fetchall()
                    rows.sort(
                        key=lambda ep: (
                            _W_RELEVANCE * id2rel.get(ep["id"], 0.0)
                            + _W_RECENCY * self._recency(ep["ts"])
                            + _W_IMPORTANCE * 0.4
                        ),
                        reverse=True,
                    )
                    return rows[:k]

        scored = []
        for ep in self._conn.execute(
            "SELECT * FROM episodes WHERE user_id=?", (user_id,)
        ).fetchall():
            s = self._score(query, ep["query"] + " " + ep["answer"], ep["ts"], 0.4, ep["embedding"])
            scored.append((s, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for s, ep in scored[:k] if s > 0.15]

    def recall(self, user_id: str, query: str, k: int = 5, budget: int = 1800) -> str:
        """
        Адаптивно собирает контекст памяти под бюджет символов (анти-bloat):
        факты о пользователе + релевантные эпизоды + выводы добавляются жадно по
        приоритету, пока не исчерпан budget. При большой памяти попадают только
        самые ценные чанки, а не всё подряд.
        """
        facts = self.get_facts(user_id)
        top_eps = self._rank_episodes(user_id, query, k)

        scored_refl = []
        for rf in self._conn.execute(
            "SELECT * FROM reflections WHERE user_id=?", (user_id,)
        ).fetchall():
            s = self._score(query, rf["insight"], rf["ts"], rf["importance"], rf["embedding"])
            scored_refl.append((s, rf))
        scored_refl.sort(key=lambda x: x[0], reverse=True)

        if not facts and not top_eps and not scored_refl:
            return "Пока нет сохранённой памяти о пользователе."

        # Кандидаты-строки в порядке приоритета: факты > выводы > эпизоды.
        sections: list[tuple[str, list[str]]] = []
        sections.append((
            "[Что я знаю о пользователе]",
            [f"- {f['key']}: {f['value']}" for f in facts],
        ))
        sections.append((
            "[Выводы из опыта]",
            [f"- {rf['insight']}" for s, rf in scored_refl if s > 0.2],
        ))
        ep_lines = []
        for ep in top_eps:
            when = time.strftime("%Y-%m-%d", time.localtime(ep["ts"]))
            tag = "✓" if ep["outcome"] == "ok" else "⚠ слабый ответ"
            ep_lines.append(f"- ({when}) «{ep['query'][:80]}» [{tag}]")
        sections.append(("[Похожие прошлые задачи]", ep_lines))

        blocks: list[str] = []
        used = 0
        for header, lines in sections:
            if used >= budget:
                break
            picked = []
            for ln in lines:
                if used + len(ln) > budget:
                    break
                picked.append(ln)
                used += len(ln) + 1
            if picked:
                blocks.append(header + "\n" + "\n".join(picked))

        return "\n\n".join(blocks) if blocks else "Пока нет релевантной памяти."

    def prune(self, max_episodes: int = 2000, max_facts: int = 300, max_reflections: int = 200) -> dict:
        """
        Защита от переполнения памяти. Эпизоды — оставляем самые свежие; факты —
        самые важные; выводы — самые свежие. Лишнее удаляем. Возвращает счётчики.
        Старые эпизоды «сжаты» в саммари/выводах, поэтому их удаление не теряет смысл.
        """
        removed = {"episodes": 0, "facts": 0, "reflections": 0}
        cur = self._conn.cursor()

        for user in [r["user_id"] for r in cur.execute("SELECT DISTINCT user_id FROM episodes").fetchall()]:
            cur.execute(
                "DELETE FROM episodes WHERE user_id=? AND id NOT IN "
                "(SELECT id FROM episodes WHERE user_id=? ORDER BY ts DESC LIMIT ?)",
                (user, user, max_episodes),
            )
            removed["episodes"] += cur.rowcount if cur.rowcount > 0 else 0

        for user in [r["user_id"] for r in cur.execute("SELECT DISTINCT user_id FROM facts").fetchall()]:
            cur.execute(
                "DELETE FROM facts WHERE user_id=? AND id NOT IN "
                "(SELECT id FROM facts WHERE user_id=? ORDER BY importance DESC, ts DESC LIMIT ?)",
                (user, user, max_facts),
            )
            removed["facts"] += cur.rowcount if cur.rowcount > 0 else 0

        for user in [r["user_id"] for r in cur.execute("SELECT DISTINCT user_id FROM reflections").fetchall()]:
            cur.execute(
                "DELETE FROM reflections WHERE user_id=? AND id NOT IN "
                "(SELECT id FROM reflections WHERE user_id=? ORDER BY ts DESC LIMIT ?)",
                (user, user, max_reflections),
            )
            removed["reflections"] += cur.rowcount if cur.rowcount > 0 else 0

        self._conn.commit()
        return removed

    def close(self) -> None:
        self._conn.close()
