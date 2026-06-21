"""chat_store — постоянный индекс ЧАТОВ (тредов) поверх чекпоинтера графа.

Чекпоинтер LangGraph (data/checkpoints.db) хранит СОСТОЯНИЕ графа по thread_id, но не
даёт пользовательской навигации: id случайные, нигде не перечислены. Этот модуль ведёт
человеко-читаемый индекс тредов (заголовок, время, ★избранное) + ПОЛНЫЙ лог сообщений
каждого треда (чтобы можно было вернуться, показать историю и СЖАТЬ её в саммари-индекс,
не теряя полную историю — идея «саммари = индекс на полную историю» из CLAUDE.md).

Лёгкий, без LLM в горячем пути: заголовок — из первого сообщения (обрезка). Сжатие
(summary) делает вызывающий код через LLM и кладёт сюда `set_summary`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Optional

_DEF_PATH = "data/chats.db"
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = os.getenv("AGENT_CHATS_DB") or _DEF_PATH
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS threads (
              thread_id  TEXT PRIMARY KEY,
              user_id    TEXT NOT NULL DEFAULT 'local',
              title      TEXT NOT NULL DEFAULT '',
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              favorite   INTEGER NOT NULL DEFAULT 0,
              summary    TEXT NOT NULL DEFAULT '',
              msg_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
              id        INTEGER PRIMARY KEY AUTOINCREMENT,
              thread_id TEXT NOT NULL,
              role      TEXT NOT NULL,
              content   TEXT NOT NULL,
              ts        REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_thread ON messages(thread_id, id);
            CREATE INDEX IF NOT EXISTS idx_thr_user ON threads(user_id, updated_at);
            """
        )
        _conn.commit()
    return _conn


def _title_from(text: str, limit: int = 60) -> str:
    t = " ".join((text or "").split())
    return (t[:limit] + "…") if len(t) > limit else t or "(без названия)"


def record_turn(thread_id: str, user_id: str, user_msg: str, assistant_msg: str) -> None:
    """Сохранить один обмен (реплика юзера + ответ) в лог треда; первый обмен задаёт
    заголовок. Дёшево, без LLM. Не роняет CLI при сбое БД (best-effort)."""
    if not thread_id:
        return
    now = time.time()
    try:
        with _lock:
            db = _db()
            row = db.execute("SELECT title, msg_count FROM threads WHERE thread_id=?",
                             (thread_id,)).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO threads(thread_id,user_id,title,created_at,updated_at,msg_count)"
                    " VALUES(?,?,?,?,?,0)",
                    (thread_id, user_id, _title_from(user_msg), now, now))
            elif not (row[0] or "").strip():
                db.execute("UPDATE threads SET title=? WHERE thread_id=?",
                           (_title_from(user_msg), thread_id))
            for role, content in (("user", user_msg), ("assistant", assistant_msg)):
                db.execute("INSERT INTO messages(thread_id,role,content,ts) VALUES(?,?,?,?)",
                           (thread_id, role, content or "", now))
            db.execute("UPDATE threads SET updated_at=?, msg_count=msg_count+2 WHERE thread_id=?",
                       (now, thread_id))
            db.commit()
    except Exception as e:  # noqa: BLE001
        print(f"[chat_store] record_turn: {type(e).__name__}: {e}")


def list_threads(user_id: str = "local", limit: int = 20, favorites_only: bool = False) -> list[dict]:
    """Недавние треды юзера (избранные первыми, затем по свежести)."""
    try:
        with _lock:
            db = _db()
            q = ("SELECT thread_id,title,created_at,updated_at,favorite,msg_count,"
                 "CASE WHEN summary<>'' THEN 1 ELSE 0 END AS compressed FROM threads WHERE user_id=?")
            args: list = [user_id]
            if favorites_only:
                q += " AND favorite=1"
            q += " ORDER BY favorite DESC, updated_at DESC LIMIT ?"
            args.append(int(limit))
            rows = db.execute(q, args).fetchall()
    except Exception as e:  # noqa: BLE001
        print(f"[chat_store] list_threads: {type(e).__name__}: {e}")
        return []
    cols = ("thread_id", "title", "created_at", "updated_at", "favorite", "msg_count", "compressed")
    return [dict(zip(cols, r)) for r in rows]


def get_messages(thread_id: str, last: Optional[int] = None) -> list[dict]:
    """Сообщения треда в формате chat_history ([{role, content}]). last=N → последние N."""
    try:
        with _lock:
            db = _db()
            if last:
                rows = db.execute(
                    "SELECT role,content FROM (SELECT id,role,content FROM messages "
                    "WHERE thread_id=? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                    (thread_id, int(last))).fetchall()
            else:
                rows = db.execute("SELECT role,content FROM messages WHERE thread_id=? ORDER BY id ASC",
                                  (thread_id,)).fetchall()
    except Exception as e:  # noqa: BLE001
        print(f"[chat_store] get_messages: {type(e).__name__}: {e}")
        return []
    return [{"role": r[0], "content": r[1]} for r in rows]


def get_thread(thread_id: str) -> Optional[dict]:
    try:
        with _lock:
            db = _db()
            r = db.execute("SELECT thread_id,user_id,title,created_at,updated_at,favorite,summary,"
                           "msg_count FROM threads WHERE thread_id=?", (thread_id,)).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not r:
        return None
    cols = ("thread_id", "user_id", "title", "created_at", "updated_at", "favorite", "summary", "msg_count")
    return dict(zip(cols, r))


def set_favorite(thread_id: str, fav: bool) -> bool:
    try:
        with _lock:
            db = _db()
            db.execute("UPDATE threads SET favorite=? WHERE thread_id=?", (1 if fav else 0, thread_id))
            db.commit()
        return True
    except Exception:  # noqa: BLE001
        return False


def toggle_favorite(thread_id: str) -> bool:
    """Переключить ★; возвращает НОВОЕ состояние."""
    t = get_thread(thread_id)
    new = not (t and t.get("favorite"))
    set_favorite(thread_id, new)
    return new


def rename(thread_id: str, title: str) -> None:
    with _lock:
        db = _db()
        db.execute("UPDATE threads SET title=? WHERE thread_id=?", (_title_from(title, 80), thread_id))
        db.commit()


def set_summary(thread_id: str, summary: str) -> None:
    """Записать сжатую версию треда (саммари-индекс на полную историю в messages)."""
    with _lock:
        db = _db()
        db.execute("UPDATE threads SET summary=? WHERE thread_id=?", (summary or "", thread_id))
        db.commit()


def delete_thread(thread_id: str) -> None:
    with _lock:
        db = _db()
        db.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
        db.execute("DELETE FROM threads WHERE thread_id=?", (thread_id,))
        db.commit()


def ensure_thread(thread_id: str, user_id: str = "local") -> None:
    """Зарегистрировать пустой тред (если ещё нет) — чтобы он был виден в /chats до первого обмена."""
    now = time.time()
    try:
        with _lock:
            db = _db()
            db.execute("INSERT OR IGNORE INTO threads(thread_id,user_id,title,created_at,updated_at)"
                       " VALUES(?,?,?,?,?)", (thread_id, user_id, "", now, now))
            db.commit()
    except Exception:  # noqa: BLE001
        pass
