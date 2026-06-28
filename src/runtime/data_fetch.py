"""
fetch_data — read-only добыча ПОЛНОГО датасета по URL (CSV/CSV.gz/JSON/bulk) → .xlsx-артефакт.

Закрывает дыру возможностей: `python_exec` засендбоксен БЕЗ сети (анти-RCE), веб-тулы чанкуют/
обрезают страницы → агент НЕ мог скачать датасет и собрать файл, только «находил источник». Этот
тул качает полный контент по прямому URL (ТОЛЬКО ЧТЕНИЕ, без выполнения кода), парсит таблицу и
пишет .xlsx в канал артефактов; модели отдаёт лишь ПРЕВЬЮ (форма/колонки/head), а не все строки —
10000+ строк в контекст не влезут (поток tool→artifact, мимо модели).

БЕЗОПАСНОСТЬ: сетевой egress на URL, который мог прийти из недоверенного веб-контента → HITL-
подтверждение (как dangerous-тулы; kind='deny' = нет канала headless → не ломаем). Read-only, без
шелла/кода → риск меньше run_bash, но egress есть. Лимиты: размер загрузки, число строк, таймаут.
"""
from __future__ import annotations

import csv as _csv
import gzip
import io
import json
import urllib.request

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

_MAX_BYTES = 60 * 1024 * 1024   # 60 МБ — кап загрузки (анти-OOM)
_MAX_ROWS = 200_000             # кап строк в .xlsx
_TIMEOUT = 40                   # сек на загрузку


class _FetchArgs(BaseModel):
    url: str = Field(description="Прямой URL на ДАННЫЕ (CSV / CSV.gz / JSON / bulk-эндпоинт) — сам "
                                 "файл данных, не страница-обёртка (напр. .csv/.json/.csv.gz).")
    filename: str = Field(default="data", description="Имя итогового файла без расширения.")
    sheet: str = Field(default="Данные", description="Имя листа .xlsx.")


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SEA-fetch_data/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:  # noqa: S310 — read-only egress, HITL-gated
        data = r.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError(f"файл больше лимита {_MAX_BYTES // (1024 * 1024)} МБ")
    return data


def _maybe_gunzip(url: str, data: bytes) -> bytes:
    if url.lower().endswith(".gz") or data[:2] == b"\x1f\x8b":  # gzip-магия
        return gzip.decompress(data)
    return data


def _looks_numeric(cell: str) -> bool:
    try:
        float(str(cell).replace(",", "."))
        return True
    except Exception:  # noqa: BLE001
        return False


def _parse_table(url: str, raw: bytes) -> tuple[list, list]:
    """(columns, rows) из CSV/JSON. Заголовок — первая строка, если она НЕ чисто числовая."""
    text = raw.decode("utf-8", errors="replace")
    stripped = text.lstrip()
    # JSON
    if url.lower().endswith(".json") or stripped[:1] in "[{":
        obj = json.loads(text)
        src = obj if isinstance(obj, list) else (obj.get("data") or obj.get("results") or [])
        if src and isinstance(src[0], dict):
            cols = list(src[0].keys())
            return cols, [[r.get(c, "") for c in cols] for r in src[:_MAX_ROWS]]
        rows = [r if isinstance(r, list) else [r] for r in src[:_MAX_ROWS]]
        ncol = max((len(r) for r in rows), default=1)
        return [f"col{i + 1}" for i in range(ncol)], rows
    # CSV (sniff разделитель)
    try:
        dialect = _csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except Exception:  # noqa: BLE001
        dialect = _csv.excel
    all_rows = [r for r in _csv.reader(io.StringIO(text), dialect) if any((c or "").strip() for c in r)]
    if not all_rows:
        raise ValueError("пусто после парсинга")
    first = all_rows[0]
    has_header = bool(first) and sum(_looks_numeric(c) for c in first) < len(first) / 2
    if has_header:
        return first, all_rows[1:_MAX_ROWS + 1]
    ncol = max((len(r) for r in all_rows), default=1)
    return [f"col{i + 1}" for i in range(ncol)], all_rows[:_MAX_ROWS]


def make_fetch_data_tool() -> StructuredTool:
    async def _run(url: str, filename: str = "data", sheet: str = "Данные") -> str:
        import asyncio

        from src.runtime import hitl, run_context
        from src.runtime.artifacts import write_table

        # HITL: сетевой egress (URL мог прийти из недоверенного веба) → подтверждение ТОЛЬКО в manual.
        # fetch_data read-only (не RCE, как run_bash) → auto-accept ЕГО ПОКРЫВАЕТ (auto_confirm), не
        # требуем полный auto. Опасное (шелл/ФС) по-прежнему гейтится строже отдельно.
        if not hitl.is_auto():
            try:
                approved, _note, kind = await hitl.confirm_rich(
                    f"fetch_data — СКАЧАТЬ данные по сети с:\n{url[:300]}\n(read-only, без выполнения кода)")
            except Exception:  # noqa: BLE001
                approved, kind = True, "deny"   # сбой канала → не ломаем (read-only)
            if not approved and kind != "deny":
                return (f"{hitl.REFUSAL_MARK}: загрузка не подтверждена пользователем. Не повторяй — "
                        "сообщи, что нужен его доступ/подтверждение к этому источнику.")
        try:
            raw = await asyncio.to_thread(_download, url)
            raw = _maybe_gunzip(url, raw)
            cols, rows = await asyncio.to_thread(_parse_table, url, raw)
        except Exception as e:  # noqa: BLE001
            return f"(не удалось скачать/распарсить {url}: {type(e).__name__}: {e})"
        if not rows:
            return f"(источник {url} не дал табличных строк)"
        run_context.mark_external_content()      # скачанное недоверенно → taint (гейт python_exec)
        meta = write_table(filename, cols, rows, sheet)
        head = " | ".join(", ".join(str(c)[:18] for c in r[:6]) for r in rows[:3])
        return (f"Файл собран и доставлен: {meta.get('name', filename)} — "
                f"{meta.get('nrows', len(rows))} строк × {len(cols)} колонок. "
                f"Колонки: {', '.join(map(str, cols[:12]))}{' …' if len(cols) > 12 else ''}. "
                f"Превью: {head[:280]}. НЕ дублируй данные текстом — файл уже у пользователя.")

    return StructuredTool.from_function(
        coroutine=_run, name="fetch_data", args_schema=_FetchArgs,
        description="DOWNLOAD a full dataset from a direct data URL (CSV / CSV.gz / JSON / bulk "
                    "endpoint) and build a downloadable .xlsx for the user. Read-only network fetch "
                    "(no code execution). Use this to ACQUIRE datasets / time-series / many rows "
                    "(weather, statistics, prices): web search/browse only FIND the source URL — this "
                    "PULLS the actual data. Returns a preview and delivers the file; the full data does "
                    "NOT pass through you. Confirm the URL is the data file itself, not a wrapper page.",
    )
