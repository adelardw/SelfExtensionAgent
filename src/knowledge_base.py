"""
База знаний пользователя: документы, которые юзер прикладывает (CLI/приложение), в
ИЕРАРХИИ папок/подпапок, с retrieval. Агент отвечает из СОБСТВЕННЫХ документов юзера
(персонализация + контекстный инжиниринг: в контекст — релевантные куски, не весь файл).

Хранение:
  data/kb/<user_id>/<папка>/<подпапка>/<файл>   — реальные файлы (иерархия юзера)
  data/kb/<user_id>/_index.db                    — SQLite-индекс чанков для retrieval

Ингест: read_file (pdf/docx/pptx/txt/md/…) → чанкинг → индекс. Поиск: BM25 (надёжно без
ключа) по чанкам юзера, опц. vector-rerank. Возвращает куски со ссылкой на документ/папку.
"""
from __future__ import annotations

import asyncio
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Optional

from .retrieval import bm25_rank

_KB_ROOT = Path("data/kb")
# Ярус 3 (временная память, CLAUDE.md): файлы, приложенные В ЭТОЙ СЕССИИ. Живут в tmp,
# чистятся в конце сессии — НЕ попадают в постоянную БЗ юзера.
_SESSION_ROOT = Path(tempfile.gettempdir()) / "agent_session_kb"


def _user_root(user_id: str) -> Path:
    p = _KB_ROOT / (re.sub(r"[^\w.-]", "_", user_id) or "default")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _lrag_dir(user_id: str) -> Path:
    return _user_root(user_id) / "_lightrag"  # рабочая директория графа LightRAG юзера


def _conn(user_id: str) -> sqlite3.Connection:
    # Чанки для BM25-ФОЛБЭКА (когда LightRAG/ключ недоступен). Граф ведёт сам LightRAG.
    c = sqlite3.connect(str(_user_root(user_id) / "_index.db"))
    c.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, folder TEXT, "
              "doc TEXT, chunk TEXT, ts REAL)")
    c.execute("CREATE INDEX IF NOT EXISTS ix_doc ON chunks(doc)")
    return c


def _chunk(text: str, target: int = 700) -> list[str]:
    """Связные куски ~target символов по абзацам/предложениям."""
    out: list[str] = []
    for para in (p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 20):
        if len(para) <= target * 2:
            out.append(para)
            continue
        buf = ""
        for s in re.split(r"(?<=[.!?])\s+", para):
            if len(buf) + len(s) > target and buf:
                out.append(buf.strip())
                buf = s
            else:
                buf = f"{buf} {s}" if buf else s
        if buf.strip():
            out.append(buf.strip())
    return out


def create_folder(user_id: str, folder: str) -> str:
    """Создать папку/подпапку в БЗ юзера (иерархия — реальные директории)."""
    rel = re.sub(r"\.\.+", ".", folder).strip("/")
    (_user_root(user_id) / rel).mkdir(parents=True, exist_ok=True)
    return rel or "(корень)"


async def add_document_async(user_id: str, src_path: str | Path, folder: str = "") -> str:
    """Добавить документ в БЗ: копия в папку (иерархия) + LightRAG-граф + BM25-чанки (floor)."""
    from .media import read_file
    from . import lightrag_engine as LR

    src = Path(src_path)
    if not src.exists():
        return f"[файл не найден: {src}]"
    rel = re.sub(r"\.\.+", ".", folder).strip("/")
    dst_dir = _user_root(user_id) / rel
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    try:
        shutil.copy2(src, dst)
    except Exception:  # noqa: BLE001 — уже там / тот же файл
        pass
    text = read_file(dst, 200_000)
    chunks = _chunk(text)
    doc_rel = str(dst.relative_to(_user_root(user_id)))
    con = _conn(user_id)
    con.execute("DELETE FROM chunks WHERE doc=?", (doc_rel,))  # BM25-фолбэк-индекс
    con.executemany("INSERT INTO chunks (folder, doc, chunk, ts) VALUES (?,?,?,?)",
                    [(rel, doc_rel, ch, time.time()) for ch in chunks])
    con.commit()
    con.close()
    # НАСТОЯЩИЙ LightRAG: граф сущностей+связей (best-effort; нет ключа → только BM25).
    graphed = await LR.insert(_lrag_dir(user_id), f"[Документ: {doc_rel}]\n{text}")
    tag = " + граф LightRAG" if graphed else " (BM25; LightRAG недоступен)"
    return f"Добавлено в БЗ: {doc_rel} ({len(chunks)} фрагментов{tag})"


def add_document(user_id: str, src_path: str | Path, folder: str = "") -> str:
    """Синхронная обёртка add_document_async (для CLI/скриптов)."""
    return asyncio.run(add_document_async(user_id, src_path, folder))


def list_kb(user_id: str) -> str:
    """Дерево папок и документов БЗ юзера + сколько фрагментов проиндексировано."""
    con = _conn(user_id)
    rows = con.execute("SELECT folder, doc, COUNT(*) FROM chunks GROUP BY doc ORDER BY folder, doc").fetchall()
    con.close()
    if not rows:
        return "База знаний пуста."
    out = []
    for folder, doc, n in rows:
        out.append(f"  {doc} ({n} фрагм.)")
    return "База знаний:\n" + "\n".join(out)


def _bm25_search(user_id: str, query: str, k: int, folder: Optional[str]) -> str:
    """BM25-ФОЛБЭК по чанкам (когда LightRAG/ключ недоступен)."""
    con = _conn(user_id)
    if folder:
        rows = con.execute("SELECT doc, chunk FROM chunks WHERE folder LIKE ?", (folder + "%",)).fetchall()
    else:
        rows = con.execute("SELECT doc, chunk FROM chunks").fetchall()
    con.close()
    if not rows:
        return "В базе знаний пользователя ничего нет (или папка пуста)."
    idx = bm25_rank([r[1] for r in rows], query, k)
    if not idx:
        return "В базе знаний не нашлось релевантного по запросу."
    return "\n\n".join(f"[{rows[i][0]}]\n{rows[i][1][:700]}" for i in idx)


async def search_kb_async(user_id: str, query: str, k: int = 5, folder: Optional[str] = None) -> str:
    """Retrieval по БЗ: СНАЧАЛА граф LightRAG (гибрид граф+вектор, multi-hop по сущностям),
    при недоступности → BM25-фолбэк. folder=None для графа (LightRAG индексирует всё)."""
    from . import lightrag_engine as LR

    if not folder:  # граф ведётся по всей БЗ юзера; folder-скоуп → BM25
        g = await LR.query(_lrag_dir(user_id), query, mode="hybrid")
        if g:
            return g
    return _bm25_search(user_id, query, k, folder)


def search_kb(user_id: str, query: str, k: int = 5, folder: Optional[str] = None) -> str:
    """Синхронная обёртка (CLI/тулы вне async-контекста)."""
    return asyncio.run(search_kb_async(user_id, query, k, folder))


def kb_has_docs(user_id: str) -> bool:
    """Есть ли у юзера хоть один документ в БЗ (для анти-bloat: тул цепляем только тогда)."""
    db = _user_root(user_id) / "_index.db"
    if not db.exists():
        return False
    try:
        con = sqlite3.connect(str(db))
        n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
        return n > 0
    except Exception:  # noqa: BLE001
        return False


def make_kb_tool(user_id: str):
    """Тул search_knowledge_base: агент ищет в ЛИЧНЫХ документах юзера (граф LightRAG)."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _Q(BaseModel):
        query: str = Field(description="Что искать в личной базе знаний пользователя")

    async def _arun(query: str) -> str:
        return await search_kb_async(user_id, query, k=5)

    return StructuredTool.from_function(
        coroutine=_arun, name="search_knowledge_base", args_schema=_Q,
        description="Search the USER'S OWN uploaded documents (their personal knowledge base, a "
                    "LightRAG entity-relation graph). Use when the question may be answered by the "
                    "user's own files, notes, or attached documents rather than the web.",
    )


# ── Ярус 3: ВРЕМЕННЫЕ файлы сессии (tmp, чистятся в конце) ─────────────────────────
def _sess_conn(session_id: str) -> sqlite3.Connection:
    root = _SESSION_ROOT / re.sub(r"[^\w.-]", "_", session_id or "default")
    root.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(root / "_index.db"))
    c.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, doc TEXT, chunk TEXT)")
    return c


def add_session_file(session_id: str, src_path: str | Path) -> str:
    """Приложить файл В ТЕКУЩЕЙ СЕССИИ (tmp): скопировать + проиндексировать. Не persist."""
    from .media import read_file

    src = Path(src_path)
    if not src.exists():
        return f"[файл не найден: {src}]"
    root = _SESSION_ROOT / re.sub(r"[^\w.-]", "_", session_id or "default")
    root.mkdir(parents=True, exist_ok=True)
    dst = root / src.name
    try:
        shutil.copy2(src, dst)
    except Exception:  # noqa: BLE001
        pass
    chunks = _chunk(read_file(dst, 200_000))
    con = _sess_conn(session_id)
    con.execute("DELETE FROM chunks WHERE doc=?", (src.name,))
    con.executemany("INSERT INTO chunks (doc, chunk) VALUES (?,?)", [(src.name, ch) for ch in chunks])
    con.commit()
    con.close()
    return f"Приложен в сессию: {src.name} ({len(chunks)} фрагментов)"


def session_has_files(session_id: str) -> bool:
    db = _SESSION_ROOT / re.sub(r"[^\w.-]", "_", session_id or "default") / "_index.db"
    if not db.exists():
        return False
    try:
        con = sqlite3.connect(str(db))
        n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
        return n > 0
    except Exception:  # noqa: BLE001
        return False


def search_session(session_id: str, query: str, k: int = 5) -> str:
    """Retrieval по файлам, приложенным в ЭТОЙ сессии."""
    con = _sess_conn(session_id)
    rows = con.execute("SELECT doc, chunk FROM chunks").fetchall()
    con.close()
    if not rows:
        return "В этой сессии файлов не приложено."
    idx = bm25_rank([r[1] for r in rows], query, k)
    if not idx:
        return "В приложенных файлах не нашлось релевантного."
    return "\n\n".join(f"[{rows[i][0]}]\n{rows[i][1][:700]}" for i in idx)


def clear_session(session_id: str) -> None:
    """Очистить временные файлы сессии (конец сессии — ярус 3 не сохраняется)."""
    root = _SESSION_ROOT / re.sub(r"[^\w.-]", "_", session_id or "default")
    shutil.rmtree(root, ignore_errors=True)


def make_session_kb_tool(session_id: str):
    """Тул search_attached_files: поиск в файлах, приложенных в текущей сессии."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _Q(BaseModel):
        query: str = Field(description="Что искать в приложенных в этой сессии файлах")

    def _run(query: str) -> str:
        return search_session(session_id, query, k=5)

    return StructuredTool.from_function(
        func=_run, name="search_attached_files", args_schema=_Q,
        description="Search files the user ATTACHED IN THIS SESSION (temporary, not the permanent "
                    "knowledge base). Use first when the user refers to a file they just attached.",
    )
