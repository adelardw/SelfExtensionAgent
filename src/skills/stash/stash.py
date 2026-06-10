"""
stash — именованные рабочие наборы структурированных данных (бюджет, таблицы,
трекеры, аналитика). Персональный «рабочий стол» агента под рутину пользователя:
фин-аналитик ведёт бюджет, разработчик — список задач, и т.п.

Хранилище — JSON-файлы в data/stashes/<name>.json (локально, без БД-зависимостей).
Каждый стэш — список строк-записей (dict). Безопасно: только локальные данные,
без device-side-effect, поэтому HITL не требуется. protected core.
"""
import csv
import io
import json
import os
import time
from pathlib import Path

from langchain_core.tools import tool

_DIR = Path(os.getenv("AGENT_STASH_DIR", "data/stashes"))


def _norm(name: str) -> str:
    return "".join(c for c in name.strip().lower().replace(" ", "_") if c.isalnum() or c in "_-") or "stash"


def _path(name: str) -> Path:
    return _DIR / f"{_norm(name)}.json"


def _resolve(name: str) -> Path:
    """
    Имя стэша для ЧТЕНИЯ: точное совпадение → оно; иначе подбираем существующий по
    подстроке (бюджет↔мой_бюджет), а если стэш всего один — берём его. Так аналитика
    не падает из-за того, что агент назвал стэш чуть иначе, чем при записи.
    """
    exact = _path(name)
    if exact.exists() or not _DIR.exists():
        return exact
    files = sorted(_DIR.glob("*.json"))
    want = _norm(name)
    for f in files:  # подстрочное совпадение в любую сторону
        if want in f.stem or f.stem in want:
            return f
    if len(files) == 1:  # единственный стэш — почти наверняка он и нужен
        return files[0]
    return exact


def _read(p: Path) -> list:
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def _resolve_write(name: str, row: dict) -> Path:
    """
    Куда ЗАПИСАТЬ запись: точное имя → оно; иначе ищем существующий стэш ТОЙ ЖЕ формы
    (совпадают колонки — бюджет↔расходы по сумма/категория), чтобы не плодить синонимы
    и не фрагментировать данные. Данные ДРУГОЙ формы (задачи) → новый стэш (точное имя).
    """
    exact = _path(name)
    if exact.exists() or not _DIR.exists():
        return exact
    want = _norm(name)
    keys = {k for k in row if k != "_ts"}
    for f in sorted(_DIR.glob("*.json")):
        if want in f.stem or f.stem in want:  # явный лексический синоним
            return f
        existing = _read(f)
        if existing and keys:  # та же ФОРМА данных → тот же стэш
            ekeys = {k for k in existing[0] if k != "_ts"}
            if len(keys & ekeys) >= max(1, len(keys) // 2):
                return f
    return exact


def _load(name: str) -> list:
    """Чтение по ТОЧНОМУ имени (для записи — не подменяем стэш)."""
    return _read(_path(name))


def _load_resolved(name: str) -> list:
    """Чтение с подбором существующего стэша (для view/aggregate/export — найти данные)."""
    return _read(_resolve(name))


def _save(name: str, rows: list) -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    _path(name).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


@tool
def stash_list() -> str:
    """List all stashes (named structured-data workspaces) with their row counts.
    Use FIRST to see what the user already tracks (budgets, tables, trackers)."""
    if not _DIR.exists():
        return "Стэшей пока нет."
    items = sorted(_DIR.glob("*.json"))
    if not items:
        return "Стэшей пока нет."
    return "Стэши:\n" + "\n".join(f"• {p.stem} ({len(_load(p.stem))} записей)" for p in items)


@tool
def stash_add(name: str, row_json: str) -> str:
    """Append a structured record (one row) to a named stash. Creates the stash if new.

    Args:
        name: Stash name (e.g. 'бюджет', 'задачи', 'расходы_июнь').
        row_json: One record as a JSON object, e.g. '{"дата":"2026-06-10","сумма":1500,"категория":"еда"}'.
    """
    try:
        row = json.loads(row_json)
        if not isinstance(row, dict):
            return "row_json должен быть JSON-объектом {ключ: значение}."
    except Exception as e:  # noqa: BLE001
        return f"Невалидный JSON: {e}"
    # Резолвим к существующему стэшу ТОЙ ЖЕ формы (бюджет vs расходы), чтобы НЕ плодить
    # синонимы и не фрагментировать данные — иначе аналитика увидит не всё.
    target = _resolve_write(name, row)
    rows = _read(target)
    row["_ts"] = time.strftime("%Y-%m-%d %H:%M")
    rows.append(row)
    _DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Добавил запись в стэш '{target.stem}' (всего {len(rows)})."


@tool
def stash_view(name: str, limit: int = 50) -> str:
    """Show records of a stash as a table (most recent first).

    Args:
        name: Stash name.
        limit: Max rows to show.
    """
    rows = _load_resolved(name)
    if not rows:
        return f"Стэш '{name}' пуст или не существует."
    cols = []
    for r in rows:
        for k in r:
            if k != "_ts" and k not in cols:
                cols.append(k)
    out = [" | ".join(cols)]
    out.append("-" * len(out[0]))
    for r in rows[-limit:]:
        out.append(" | ".join(str(r.get(c, "")) for c in cols))
    return f"Стэш '{name}' ({len(rows)} записей):\n" + "\n".join(out)


@tool
def stash_aggregate(name: str, field: str, op: str = "sum", group_by: str = "") -> str:
    """Compute a quick aggregate over a numeric field (analytics on a stash).

    Args:
        name: Stash name.
        field: Numeric field to aggregate (e.g. 'сумма').
        op: 'sum' | 'avg' | 'min' | 'max' | 'count'.
        group_by: Optional field to group by (e.g. 'категория').
    """
    rows = _load_resolved(name)
    if not rows:
        return f"Стэш '{name}' пуст."

    def _num(v):
        try:
            return float(str(v).replace(",", ".").replace(" ", ""))
        except Exception:  # noqa: BLE001
            return None

    def _agg(vals: list[float]) -> float:
        if op == "count":
            return len(vals)
        if not vals:
            return 0.0
        return {"sum": sum(vals), "avg": sum(vals) / len(vals),
                "min": min(vals), "max": max(vals)}.get(op, sum(vals))

    if group_by:
        groups: dict = {}
        for r in rows:
            g = str(r.get(group_by, "—"))
            v = _num(r.get(field))
            groups.setdefault(g, []).append(v if v is not None else 0.0)
        lines = [f"{g}: {_agg(vs):.2f}" for g, vs in sorted(groups.items(), key=lambda x: -_agg(x[1]))]
        return f"{op}({field}) по '{group_by}' в '{name}':\n" + "\n".join(lines)

    vals = [v for v in (_num(r.get(field)) for r in rows) if v is not None]
    return f"{op}({field}) в '{name}' = {_agg(vals):.2f} (по {len(vals)} записям)"


@tool
def stash_export_csv(name: str) -> str:
    """Export a stash as CSV text (for pasting into a spreadsheet).

    Args:
        name: Stash name.
    """
    rows = _load_resolved(name)
    if not rows:
        return f"Стэш '{name}' пуст."
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return f"CSV стэша '{name}':\n{buf.getvalue()}"
