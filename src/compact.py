"""
Сжатие контекста CLI (`/compact`) + статус-бар занятости контекстного окна + `COMPACT.md`.

Идея (запрос юзера): пользователь САМ решает, когда сжимать. Он видит статус-бар — сколько
текста накопилось в контекстном окне за сессию (кумулятивно), с подсветкой порогов
128k/256k/512k/1M. На 1M (предел модели) — авто-сжатие. `/compact` сворачивает накопленное в
`COMPACT.md` (репрезентативно: что ключевое/инсайты, какие MCP подключил юзер и какие выбрал
агент, какие SKILLS юзера и агента, что за проект) — КУМУЛЯТИВНО: каждое сжатие ссылается на
прошлые. На основе `COMPACT.md` агент может перестроить `SEA.md`/`develop.md` (полная сверка).
Последний скоуп (свежие сообщения) сохраняется, чтобы юзер и агент не теряли нить.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

CONTEXT_MAX = 1_000_000                     # предел контекстного окна (gemini 1M) → авто-сжатие
_THRESHOLDS = (128_000, 256_000, 512_000, 1_000_000)
_KEEP_LAST_SCOPE = 6                        # сколько свежих сообщений сохранить после сжатия


def _root() -> Path:
    return Path(os.getenv("AGENT_PROJECT_ROOT") or Path.cwd())


def compact_path() -> Path:
    return _root() / "COMPACT.md"


def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов (~4 символа/токен). Для статус-бара, не для биллинга."""
    return len(text or "") // 4


def history_tokens(chat_history: list) -> int:
    return estimate_tokens("".join(str(m.get("content", "")) for m in (chat_history or [])))


def band_label(tokens: int) -> str:
    """Метка порога: 128k / 128k+ / 256k / 256k+ / 512k / 512k+ / 1m (как просил юзер)."""
    if tokens >= 1_000_000:
        return "1m"
    if tokens >= 512_000:
        return "512k+"
    if tokens >= 256_000:
        return "256k+" if tokens > 256_000 else "256k"
    if tokens >= 128_000:
        return "128k+" if tokens > 128_000 else "128k"
    return f"{tokens // 1000}k"


def _band_color(tokens: int) -> str:
    if tokens >= 900_000:
        return "red"
    if tokens >= 512_000:
        return "yellow"
    if tokens >= 256_000:
        return "cyan"
    return "green"


def context_status(tokens: int, width: int = 16) -> str:
    """Rich-markup строка статус-бара занятости контекста (для REPL)."""
    frac = min(1.0, tokens / CONTEXT_MAX)
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    color = _band_color(tokens)
    return (f"[{color}]Контекст: {band_label(tokens)} / 1M  [{bar}]  {frac:.0%}[/]"
            f"  [dim](/compact — сжать вручную)[/]")


def should_auto_compact(tokens: int) -> bool:
    return tokens >= CONTEXT_MAX


def read_compact() -> str:
    p = compact_path()
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _compaction_count() -> int:
    return read_compact().count("\n## Сжатие ")


def append_compact(summary: str) -> int:
    """Добавить новое сжатие в COMPACT.md КУМУЛЯТИВНО (со ссылкой на прошлые). Возвращает номер."""
    p = compact_path()
    prior = _compaction_count()
    n = prior + 1
    ref = "" if prior == 0 else f" (продолжает сжатия 1–{prior})"
    block = (f"\n## Сжатие {n} · {time.strftime('%Y-%m-%d %H:%M')}{ref}\n\n{summary.strip()}\n")
    header = "" if p.exists() else (
        "# COMPACT.md — кумулятивный сжатый контекст сессий\n\n"
        "Сжатые срезы контекста (репрезентативно: ключевое, инсайты, подключённые MCP/навыки, "
        "суть проекта). Каждое сжатие ссылается на прошлые. Из этого можно перестроить "
        "SEA.md/develop.md.\n")
    with p.open("a", encoding="utf-8") as f:
        if header:
            f.write(header)
        f.write(block)
    return n


def gather_meta() -> dict:
    """Контекст для сжатия: суть проекта (SEA.md), MCP/навыки юзера (MCP.md/SKILL.md). Best-effort.
    «Что выбрал агент» извлекает LLM из самого диалога."""
    meta: dict = {}
    try:
        from . import context_files as cf
        sea = cf.read("SEA.md")
        meta["project"] = sea[:1500] if sea else ""
        meta["user_mcp"] = [s.get("name") for s in cf.mcp_servers()]
        meta["user_skills_present"] = bool(cf.read("SKILL.md"))
    except Exception:  # noqa: BLE001
        pass
    try:
        from .mcp_client import TRUSTED_SERVERS
        meta["trusted_mcp"] = list(TRUSTED_SERVERS)
    except Exception:  # noqa: BLE001
        pass
    return meta


def build_compact_messages(convo: str, prior_compact: str, meta: dict):
    """Системное+human сообщение для LLM-сжатия в репрезентативный срез."""
    from langchain_core.messages import SystemMessage, HumanMessage

    sys = (
        "Сожми сессию в РЕПРЕЗЕНТАТИВНЫЙ срез для COMPACT.md (память-индекс, не дамп). Структура:\n"
        "### Ключевое и инсайты — что решили/поняли, важные развилки, договорённости.\n"
        "### Проект — что это, что внутри (стек/структура), текущее состояние/цели.\n"
        "### MCP — подключённые сервера: какие задал ПОЛЬЗОВАТЕЛЬ (из MCP.md) и какие выбрал/нашёл "
        "АГЕНТ по ходу (извлеки из диалога).\n"
        "### Навыки — какие навыки от ПОЛЬЗОВАТЕЛЯ (SKILL.md) и какие выбрал/создал АГЕНТ.\n"
        "### Открытые задачи — что не доделано, что дальше.\n"
        "Плотно, по делу, на языке диалога. Если есть прошлые сжатия — НЕ повторяй их, добавляй "
        "новое и ссылайся («см. сжатие N»)."
    )
    parts = []
    if meta.get("project"):
        parts.append(f"[Проект (SEA.md)]\n{meta['project']}")
    if meta.get("user_mcp"):
        parts.append(f"[MCP от пользователя (MCP.md)]: {', '.join(filter(None, meta['user_mcp']))}")
    if meta.get("trusted_mcp"):
        parts.append(f"[Доверенные MCP сейчас]: {', '.join(meta['trusted_mcp'])}")
    if prior_compact:
        parts.append(f"[Прошлые сжатия (НЕ повторять, ссылаться)]\n{prior_compact[-4000:]}")
    parts.append(f"[Диалог сессии]\n{convo[:16000]}")
    return [SystemMessage(content=sys), HumanMessage(content="\n\n".join(parts))]


def build_sea_rebuild_messages(compact_text: str, current_sea: str, repo_scan: str):
    """LLM-сообщения для /sync: перестроить SEA.md СВЕРЯЯ с накопленным COMPACT.md + скан репо."""
    from langchain_core.messages import SystemMessage, HumanMessage

    sys = (
        "Перестрой SEA.md — инструкции и карту проекта для агента — СВЕРЯЯ с накопленным контекстом. "
        "Сохрани полезные РУЧНЫЕ инструкции из текущего SEA.md, обнови карту репо (стек/структура/"
        "команды) по свежему скану и добавь актуальное СОСТОЯНИЕ/решения/подключённые MCP и навыки из "
        "COMPACT.md. Без воды, по делу, markdown. Верни ТОЛЬКО содержимое нового SEA.md (без пояснений)."
    )
    human = (f"[Текущий SEA.md]\n{current_sea[:4000]}\n\n"
             f"[Свежий скан репозитория]\n{repo_scan[:2500]}\n\n"
             f"[COMPACT.md — накопленный сжатый контекст]\n{compact_text[:8000]}")
    return [SystemMessage(content=sys), HumanMessage(content=human)]
