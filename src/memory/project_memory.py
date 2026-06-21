"""
Проектный ярус памяти (MEMORY.md) — порт паттерна memdir из backup (#2).

Три яруса памяти агента: SQLite = ГЛОБАЛЬНАЯ (кросс-проект), ЭТОТ модуль = ЛОКАЛЬНАЯ/ПРОЕКТНАЯ
(курируемая человекочитаемая), session = ВРЕМЕННАЯ. Здесь — файловая память проекта:
  data/project_memory/MEMORY.md   — индекс (всегда грузится в контекст: одна строка на заметку)
  data/project_memory/<slug>.md   — типизированная заметка с frontmatter (type: user|feedback|
                                    project|reference) + телом; на recall подмешиваются ≤k релевантных.

АДДИТИВНО И БЕЗОПАСНО: пустая папка → block() возвращает "" → recall_node ничего не меняет
(ноль изменения поведения по умолчанию). Таксономия (как в backup): хранить только то, что НЕ
выводится из кода/git/истории — профиль юзера, его правила-фидбек, цели проекта, ссылки.
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

from src.search.retrieval import bm25_rank

_ROOT = Path("data/project_memory")        # типизированные заметки (агент сюда пишет)
_INDEX = Path("MEMORY.md")                  # индекс — КОРНЕВОЙ файл-конвенция (как SEA.md/SKILL.md)
_TYPES = ("user", "feedback", "project", "reference")
_SLUG_RE = re.compile(r"[^\w]+", re.UNICODE)  # \w с UNICODE сохраняет кириллицу (проект русский)

# Запись атомарна + лок на RMW индекса (CON-3): make_project_memory_tool пишет заметки из
# step_executor; параллельные прогоны (мульти-клиент) на голом write_text давали read-modify-write
# гонку MEMORY.md (потеря строки-указателя) и torn-read при block(). temp→fsync→os.replace + Lock.
_LOCK = threading.Lock()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _memory_files() -> list[Path]:
    if not _ROOT.exists():
        return []
    return sorted(p for p in _ROOT.glob("*.md") if p.name != "MEMORY.md")


def _parse(path: Path) -> tuple[dict, str]:
    """Разбирает .md: простой frontmatter между --- ... --- + тело. Без внешних зависимостей."""
    text = path.read_text(encoding="utf-8")
    fm: dict = {}
    body = text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip()
        body = m.group(2)
    return fm, body.strip()


def index_text() -> str:
    """Содержимое MEMORY.md (всегда-загружаемый индекс) или '' если его нет."""
    return _INDEX.read_text(encoding="utf-8").strip() if _INDEX.exists() else ""


def recall(query: str, k: int = 5) -> str:
    """≤k релевантных заметок под запрос (BM25 по description+тело). '' если заметок нет."""
    files = _memory_files()
    if not files:
        return ""
    docs, metas = [], []
    for p in files:
        fm, body = _parse(p)
        docs.append(f"{fm.get('description', '')} {body}")
        metas.append((fm, body))
    out = []
    for i in bm25_rank(docs, query, k):
        fm, body = metas[i]
        title = fm.get("name") or fm.get("description") or "заметка"
        tp = fm.get("type", "")
        out.append(f"• [{tp}] {title}: {body[:400]}")
    return "\n".join(out)


def block(query: str, k: int = 5) -> str:
    """Полный блок проектной памяти для memory_context: индекс (всегда) + релевантные (≤k).
    '' если папки/заметок нет → вызывающий код не добавляет ничего (аддитивно)."""
    idx, rel = index_text(), recall(query, k)
    parts = []
    if idx:
        parts.append("[Проектная память — индекс]\n" + idx)
    if rel:
        parts.append("[Релевантные проектные заметки]\n" + rel)
    return "\n\n".join(parts)


def _slug(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower()).strip("-") or "memory"


def add(name: str, description: str, body: str, mtype: str = "project") -> Path:
    """Записать типизированную заметку + строку-указатель в MEMORY.md. Хранить только то,
    что НЕ выводится из кода/git (профиль/фидбек/цели/ссылки)."""
    if mtype not in _TYPES:
        mtype = "project"
    _ROOT.mkdir(parents=True, exist_ok=True)
    slug = _slug(name)
    path = _ROOT / f"{slug}.md"
    _atomic_write_text(
        path,
        f"---\nname: {name}\ndescription: {description}\ntype: {mtype}\n---\n\n{body.strip()}\n",
    )
    # Указатель в индекс ПОД ЛОКОМ: read-existing + write — одна критсекция (иначе параллельные
    # add() теряют строки друг друга), запись атомарна (temp→os.replace, без torn-read у block()).
    line = f"- [{name}]({slug}.md) — {description}"
    with _LOCK:
        existing = index_text()
        if f"({slug}.md)" not in existing:
            header = existing if existing else "# Project Memory Index"
            _atomic_write_text(_INDEX, header + "\n" + line + "\n")
    return path


def make_project_memory_tool():
    """Инструмент: агент сам сохраняет проектную заметку (профиль/фидбек/цель/ссылку)."""
    from langchain_core.tools import StructuredTool

    def _save(name: str, description: str, body: str, type: str = "project") -> str:  # noqa: A002
        p = add(name, description, body, type)
        return f"Проектная заметка сохранена: {p.name}"

    return StructuredTool.from_function(
        func=_save,
        name="save_project_memory",
        description=(
            "Сохранить ПРОЕКТНУЮ заметку (переживает сессии): профиль пользователя, его "
            "правило-предпочтение (feedback), цель/ограничение проекта или ссылку (reference). "
            "Только то, что НЕ выводится из кода/истории. type: user|feedback|project|reference."
        ),
    )
