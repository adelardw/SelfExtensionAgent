"""
ParamStore — реестр обучаемых параметров графа (мульти-артефактный).

Граф агента = дифференцируемая программа. У каждой ноды есть «параметры», которые
обновляет self-learning, НЕ трогая исходники (обратимо, версионируется):
  • prompt    — системный промпт ноды (override);
  • fewshots  — собранные удачные примеры (input→output) для few-shot инъекции;
  • tools     — переопределённые описания инструментов.

Few-shots — самый дешёвый канал генерализации: успешные прогоны копятся как
примеры и подмешиваются в промпт. Чем больше накоплено — тем устойчивее поведение
(без единого доп. LLM-вызова на сбор).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PARAMS_FILE = Path("data/params.json")

MAX_FEWSHOTS = 8  # потолок примеров на ноду (анти-переполнение)


def _load() -> dict:
    if PARAMS_FILE.exists():
        try:
            return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save(data: dict) -> None:
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── prompts (override) ───────────────────────────────────────────────

def get_prompt(role: str, default: str) -> str:
    return _load().get(role, {}).get("text", default)


def save_override(role: str, text: str, rationale: str = "") -> int:
    data = _load()
    node = data.setdefault(role, {})
    node["version"] = node.get("version", 0) + 1
    node["text"] = text
    node["rationale"] = rationale
    node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save(data)
    return node["version"]


def revert(role: str) -> bool:
    data = _load()
    if role in data and "text" in data[role]:
        data[role].pop("text", None)
        _save(data)
        return True
    return False


def list_overrides() -> dict:
    return {
        r: {"version": v.get("version"), "fewshots": len(v.get("fewshots", []))}
        for r, v in _load().items()
        if "text" in v or v.get("fewshots")
    }


# ── few-shots (forward-харвест удачных примеров) ─────────────────────

def add_fewshot(role: str, query: str, answer: str, score: float) -> None:
    """Добавляет удачный пример к ноде; держит топ-MAX_FEWSHOTS по score."""
    data = _load()
    node = data.setdefault(role, {})
    shots = node.get("fewshots", [])
    # дедуп по началу запроса
    qkey = query.strip()[:60].lower()
    shots = [s for s in shots if s["query"].strip()[:60].lower() != qkey]
    shots.append({"query": query[:400], "answer": answer[:600], "score": round(score, 3), "ts": time.time()})
    shots.sort(key=lambda s: (s["score"], s["ts"]), reverse=True)
    node["fewshots"] = shots[:MAX_FEWSHOTS]
    _save(data)


def get_fewshots(role: str, k: int = 3) -> list[dict]:
    return _load().get(role, {}).get("fewshots", [])[:k]


# ── per-user few-shots (векторизованное самоулучшение: учимся на удачных ответах
#    КОНКРЕТНОМУ пользователю и применяем ИМЕННО ему) ──────────────────────────
# Отдельный файл, чтобы не раздувать глобальный params.json и легко чистить (LRU).
# Промпт-оверрайды остаются ГЛОБАЛЬНЫМИ (per-user их крутить — оверфит/раздувание);
# персонализируется самый дешёвый и обратимый канал — примеры.

USER_FEWSHOTS_FILE = Path("data/user_fewshots.json")
MAX_USERS = 200  # потолок числа пользователей в сторе (LRU-вытеснение)


def _load_users() -> dict:
    if USER_FEWSHOTS_FILE.exists():
        try:
            return json.loads(USER_FEWSHOTS_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_users(data: dict) -> None:
    USER_FEWSHOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_FEWSHOTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def add_user_fewshot(user_id: str, role: str, query: str, answer: str, score: float) -> None:
    """Удачный пример, привязанный к пользователю (приоритетнее глобальных при инъекции)."""
    if not user_id:
        return add_fewshot(role, query, answer, score)
    data = _load_users()
    user = data.setdefault(user_id, {"_ts": 0.0})
    user["_ts"] = time.time()  # для LRU
    shots = user.get(role, [])
    qkey = query.strip()[:60].lower()
    shots = [s for s in shots if s["query"].strip()[:60].lower() != qkey]
    shots.append({"query": query[:400], "answer": answer[:600], "score": round(score, 3), "ts": time.time()})
    shots.sort(key=lambda s: (s["score"], s["ts"]), reverse=True)
    user[role] = shots[:MAX_FEWSHOTS]
    # LRU-вытеснение, чтобы стор не рос бесконечно (важно для сервера с многими юзерами)
    if len(data) > MAX_USERS:
        oldest = sorted(data.items(), key=lambda kv: kv[1].get("_ts", 0.0))[: len(data) - MAX_USERS]
        for uid, _ in oldest:
            data.pop(uid, None)
    _save_users(data)


def get_user_fewshots(user_id: str, role: str, k: int = 3) -> list[dict]:
    return (_load_users().get(user_id, {}).get(role, []) if user_id else [])[:k]


def format_fewshots(role: str, k: int = 3, user_id: str = "") -> str:
    """
    Примеры для инъекции: сначала ПЕРСОНАЛЬНЫЕ (этого пользователя), затем глобальные
    добивают до k. Так агент сначала опирается на то, что заходило именно этому человеку.
    """
    shots = list(get_user_fewshots(user_id, role, k))
    seen = {s["query"].strip()[:60].lower() for s in shots}
    for s in get_fewshots(role, k):
        if len(shots) >= k:
            break
        if s["query"].strip()[:60].lower() not in seen:
            shots.append(s)
    if not shots:
        return "Примеров пока нет."
    return "\n\n".join(f"Пример {i+1}:\nЗапрос: {s['query']}\nХороший ответ: {s['answer']}" for i, s in enumerate(shots))


# ── tool descriptions ────────────────────────────────────────────────

def get_tool_desc(tool_name: str, default: str) -> str:
    return _load().get("__tools__", {}).get(tool_name, {}).get("text", default)


def save_tool_desc(tool_name: str, text: str) -> None:
    data = _load()
    tools = data.setdefault("__tools__", {})
    tools[tool_name] = {"text": text, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save(data)
