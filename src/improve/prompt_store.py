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
import os
import tempfile
import threading
import time
from pathlib import Path

PARAMS_FILE = Path("data/params.json")

# Сериализует запись стора: add_fewshot/add_user_fewshot зовутся из ФОНОВЫХ reflect-потоков
# (_post_reflect) параллельно по запросам + per-user backward-воркер. Раньше голый write_text без
# лока/atomic → гонка read-modify-write, а полузаписанный файл _load глотал в {} → МОЛЧА обнулял ВСЕ
# выученные параметры/few-shots (тот же класс, что 2c в intent.py). Lock + temp→fsync→os.replace.
_SAVE_LOCK = threading.Lock()


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    with _SAVE_LOCK:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)             # атомарный rename: читатель видит старый ИЛИ новый ЦЕЛЫЙ файл
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

MAX_FEWSHOTS = 8  # потолок примеров на ноду (анти-переполнение)


def _load() -> dict:
    if PARAMS_FILE.exists():
        try:
            return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save(data: dict) -> None:
    _atomic_write_json(PARAMS_FILE, data)      # лок + atomic (см. _atomic_write_json)


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

def add_fewshot(role: str, query: str, answer: str, score: float, kind: str = "example") -> None:
    """Добавляет удачный пример к ноде; держит топ-MAX_FEWSHOTS по score. kind='example' —
    пара запрос→ответ/режим (forward-харвест); kind='lesson' — урок-проза из backward (рендерится
    ОТДЕЛЬНО, не как «Хороший ответ», чтобы не путать классификатор — долг ревью B6)."""
    data = _load()
    node = data.setdefault(role, {})
    shots = node.get("fewshots", [])
    # дедуп по началу запроса
    qkey = query.strip()[:60].lower()
    shots = [s for s in shots if s["query"].strip()[:60].lower() != qkey]
    shots.append({"query": query[:400], "answer": answer[:600], "score": round(score, 3),
                  "ts": time.time(), "kind": kind})
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
    _atomic_write_json(USER_FEWSHOTS_FILE, data)   # лок + atomic (тот же _SAVE_LOCK)


def add_user_fewshot(user_id: str, role: str, query: str, answer: str, score: float,
                     kind: str = "example") -> None:
    """Удачный пример/урок, привязанный к пользователю (приоритетнее глобальных при инъекции).
    kind='lesson' (backward-урок) рендерится отдельным блоком, не как mode-метка (B6)."""
    if not user_id:
        return add_fewshot(role, query, answer, score, kind)
    data = _load_users()
    user = data.setdefault(user_id, {"_ts": 0.0})
    user["_ts"] = time.time()  # для LRU
    shots = user.get(role, [])
    qkey = query.strip()[:60].lower()
    shots = [s for s in shots if s["query"].strip()[:60].lower() != qkey]
    shots.append({"query": query[:400], "answer": answer[:600], "score": round(score, 3),
                  "ts": time.time(), "kind": kind})
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


# ДВУХЪЯРУСНЫЕ few-shots (CLAUDE.md стр.91): встроенный НЕИЗМЕНЯЕМЫЙ baseline (пол
# качества, есть всегда, даже у нового юзера без истории) + поверх ОБУЧАЕМЫЕ
# персональные/глобальные. Baseline — общие ПРИНЦИПЫ маршрутизации, не eval-фразы
# (анти-оверфит): иллюстрируют классы, а не конкретные сценарии.
BASE_FEWSHOTS: dict[str, list[dict]] = {
    "reflexion": [
        {"query": "приветствие или простой вопрос из памяти", "answer": "fast"},
        {"query": "задача на расчёт/логику, где все данные в самом запросе", "answer": "reason"},
        {"query": "нужны свежие/внешние данные или действие с устройством/файлом", "answer": "deliberate"},
        # Вывод ДОЛЖЕН быть ФАЙЛОМ (excel/csv/таблица-файлом/выгрузка/отчёт-в-файл): даже если данные
        # «известны», нельзя ответить текстом — надо СОБРАТЬ и СОЗДАТЬ файл инструментом → deliberate,
        # НЕ fast/act. (Живой eval: на «собери в excel инфляцию» модель уходила в act и отказывалась.)
        {"query": "собери/выгрузи данные в excel или csv файл, дай таблицу файлом, отчёт файлом — нужно создать и отдать файл", "answer": "deliberate"},
        {"query": "размытый запрос без конкретики (что именно нужно — неясно)", "answer": "clarify"},
    ],
}


def _shot_sim(query: str, shot_q: str) -> float:
    """Дешёвая лексическая близость запроса к примеру (Jaccard по токенам len>2) — без сети.
    Локальная (не импортируем memory.store, чтобы не плодить связи improve↔memory)."""
    ta = {t for t in __import__("re").findall(r"[\wа-яё]+", (query or "").lower()) if len(t) > 2}
    tb = {t for t in __import__("re").findall(r"[\wа-яё]+", (shot_q or "").lower()) if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def format_fewshots(role: str, k: int = 3, user_id: str = "", query: str = "") -> str:
    """
    Примеры для инъекции, по приоритету: ПЕРСОНАЛЬНЫЕ (этого юзера) → глобальные обучаемые →
    встроенный baseline (неизменяемый пол). Так у нового юзера без истории всё равно есть
    базовые принципы, а с опытом сверху ложится то, что заходит ИМЕННО ему.

    query задан → внутри каждого яруса примеры ранжируются ПО ПОХОЖЕСТИ к запросу (а не по
    голому score): «такой запрос → такой ответ/режим» — kNN-классификатор маршрутизации
    (Thread 3a), не статический приор. Персональные остаются приоритетнее глобальных.
    """
    base = BASE_FEWSHOTS.get(role, [])
    # Резервируем слоты под baseline (гарантированный ПОЛ), чтобы обучаемые/глобальные —
    # которые могут быть зашумлены (в т.ч. eval-запросами) — не вытеснили принципы полностью.
    base_quota = min(len(base), max(1, k // 2)) if base else 0
    learn_k = k - base_quota

    def _ranked(pool: list[dict]) -> list[dict]:
        # query → similarity-retrieved (kNN по лексике); иначе порядок score (как было).
        return sorted(pool, key=lambda s: _shot_sim(query, s["query"]), reverse=True) if query else pool

    shots: list[dict] = []
    seen: set[str] = set()

    def _add(pool: list[dict], target: int) -> None:
        for s in pool:
            if len(shots) >= target:
                return
            key = s["query"].strip()[:60].lower()
            if key not in seen:
                shots.append(s)
                seen.add(key)

    # Берём ПОЛНЫЕ пулы (до MAX_FEWSHOTS) и ранжируем по похожести — иначе similarity видела бы
    # лишь топ-k по score. Персональные всё равно идут первыми (персонализация > усреднение).
    _add(_ranked(get_user_fewshots(user_id, role, MAX_FEWSHOTS)), learn_k)  # персональные
    _add(_ranked(get_fewshots(role, MAX_FEWSHOTS)), learn_k)                # глобальные обучаемые
    _add(base, k)                                                          # baseline — гарантированный пол
    if not shots:
        return "Примеров пока нет."

    # B6: уроки-проза (backward) рендерятся ОТДЕЛЬНЫМ блоком, НЕ как «Хороший ответ: <режим>» —
    # иначе классификатор маршрутизации видел бы в слоте ответа то метку (deliberate), то абзац.
    examples = [s for s in shots if s.get("kind") != "lesson"]
    lessons = [s for s in shots if s.get("kind") == "lesson"]
    blocks = [f"Пример {i+1}:\nЗапрос: {s['query']}\nХороший ответ: {s['answer']}"
              for i, s in enumerate(examples)]
    if lessons:
        blocks.append("Уроки (чего избегать на похожих запросах):\n" +
                      "\n".join(f"- При «{s['query']}»: {s['answer']}" for s in lessons))
    return "\n\n".join(blocks)


# ── tool descriptions ────────────────────────────────────────────────

def get_tool_desc(tool_name: str, default: str) -> str:
    return _load().get("__tools__", {}).get(tool_name, {}).get("text", default)


def save_tool_desc(tool_name: str, text: str) -> None:
    data = _load()
    tools = data.setdefault("__tools__", {})
    tools[tool_name] = {"text": text, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save(data)
