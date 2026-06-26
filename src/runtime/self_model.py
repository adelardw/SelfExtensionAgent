"""
Раздел M — само-моделирующий агент (ДЕТЕРМИНИРОВАННО, без LLM/сети → дёшево, тестируемо без API).

Содержит несущую часть M0 + scaffolding под M1-M5:
  • env-id — адресация «проекта» (среды): изоляция по умолчанию, склейка сессий только по
    УСТОЙЧИВОМУ якорю (код=путь репо, чат=персист thread_id, веб=эфемерный). Ложная склейка
    вреднее ложного раздела (травит overlay/личность чужим контекстом) → склоняемся к изоляции.
  • self-model — компактный автопортрет из 5 граней (структура / способности / знания /
    собеседник / личность) + overlay среды + инкрементальный diff. Composer читает как ПРИОР.
  • личностная память — кросс-проектный АБСТРАКТНЫЙ дистиллят (НЕ сырые факты: факт приватного
    код-проекта не должен всплыть в случайном веб-чате; сырьё/PII отсекается).
  • overlay (Type 1) со СТЕНОЙ к Type 2 — оптимизируемые промпты СРЕДЫ живут в своём namespace
    `env:<id>`; промпты когнитивных нод (Type 2) физически вне досягаемости (отдельный store +
    guard). База заморожена, оптимизируется только поведенческая политика среды.
  • session-commit + diff — запись «что сделано» в проекте и инкрементальный пересмотр (что
    изменилось с прошлой сессии В ЭТОМ проекте), чтобы интроспекция сама амортизировалась.

Всё включается ТОЛЬКО из composer (флаг experimental.composer) — живой граф не затрагивается.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

# ── env-id: изоляция по умолчанию, склейка только по устойчивому якорю ──────────────────────
# Асимметрия цены ошибки: ложно РАЗДЕЛИТЬ = потерять немного амортизации (холодный старт);
# ложно СКЛЕИТЬ = отравить overlay/личность чужим контекстом (галлюцинации). Раз склейка
# вреднее — без устойчивого якоря НЕ склеиваем, а заводим эфемерный изолированный проект.

def resolve_env_id(surface: str, anchor: str = "", run_id: str = "") -> str:
    """Стабильный id «проекта» (среды). anchor — устойчивый якорь среды (путь репо / thread_id /
    явный project-key). Без якоря → эфемерный изолированный проект (по run_id), не склеиваем."""
    surface = (surface or "chat").strip().lower()
    anchor = (anchor or "").strip()
    if anchor:
        if surface == "code":
            # путь репо может быть длинным/секретным → хэшируем (стабильно, не светим путь)
            return "code:" + hashlib.sha1(anchor.encode("utf-8")).hexdigest()[:16]
        return f"{surface}:{anchor}"
    return "ephemeral:" + (run_id or "default")


def _safe_name(env_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", env_id)[:80]


# ── Type 1 (overlay среды, оптимизируется) vs Type 2 (база, заморожено) ─────────────────────
# СТЕНА: overlay-оптимизатор работает ТОЛЬКО в своём store по env_id. Промпты когнитивных нод
# (Type 2) — отдельный механизм (improve/prompt_store + policy optimize_core_prompts:false),
# сюда физически не попадают. Guard ниже отвергает попытку записать Type-2 роль как overlay.
TYPE2_FROZEN = frozenset({
    "goal", "reflexion", "decompose", "fast_answer", "reason", "step_execution",
    "review", "clarify_gate", "router", "synthesize", "validation", "act", "verify", "finalize",
})

OVERLAY_FILE = Path("data/env_overlays.json")


def is_type1(key: str) -> bool:
    """Оптимизировать поведенческой политикой можно ТОЛЬКО overlay среды (env:<id>)."""
    return isinstance(key, str) and key.startswith("env:") and key[4:] not in TYPE2_FROZEN


def _load_overlays() -> dict:
    if OVERLAY_FILE.exists():
        try:
            return json.loads(OVERLAY_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_overlays(data: dict) -> None:
    OVERLAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OVERLAY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OVERLAY_FILE)


def get_overlay(env_id: str) -> str:
    """Выученная ПОВЕДЕНЧЕСКАЯ ПОЛИТИКА среды (не факты, не структура). Пусто, если не училась."""
    return _load_overlays().get(env_id, {}).get("text", "")


def save_overlay(env_id: str, text: str, rationale: str = "") -> int:
    """Сохранить/обновить overlay среды. СТЕНА: env_id не может быть Type-2 ролью."""
    if env_id in TYPE2_FROZEN:
        raise ValueError(f"Type-2 заморожен: '{env_id}' нельзя оптимизировать как overlay среды")
    data = _load_overlays()
    node = data.setdefault(env_id, {})
    node["version"] = node.get("version", 0) + 1
    node["text"] = text
    node["rationale"] = rationale          # «почему» — версионируем правку (дисциплина ouroboros)
    node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_overlays(data)
    return node["version"]


def revert_overlay(env_id: str) -> bool:
    """Откат overlay среды (accept/revert — самоправка всегда обратима)."""
    data = _load_overlays()
    if env_id in data and "text" in data[env_id]:
        data[env_id].pop("text", None)
        _save_overlays(data)
        return True
    return False


# ── личностная память: кросс-проектный АБСТРАКТНЫЙ дистиллят ─────────────────────────────────
# Единственный легальный кросс-проектный канал. Дисциплина утечки: только короткие абстрактные
# строки политики/идентичности; сырьё/PII (url/почта/длинные числа) НЕ принимается.
PERSONALITY_FILE = Path("data/personality.json")
MAX_PERSONALITY = 40
_RAW_FACT = re.compile(r"https?://|\b[\w.+-]+@[\w-]+\.[\w.]+|\d{4,}")


def add_personality_note(text: str, source_env: str = "") -> bool:
    """Добавить абстрактную черту в личность. Возвращает False, если строка похожа на сырой факт/PII
    (тогда ей место в проектной памяти, не в кросс-проектной личности)."""
    text = (text or "").strip()
    if not text or _RAW_FACT.search(text):
        return False
    data = _load_personality()
    notes = data.get("notes", [])
    key = text[:80].lower()
    notes = [n for n in notes if n["text"][:80].lower() != key]   # дедуп
    notes.append({"text": text[:300], "source": source_env, "ts": time.time()})
    notes.sort(key=lambda n: n["ts"], reverse=True)
    data["notes"] = notes[:MAX_PERSONALITY]
    _save_personality(data)
    return True


def format_personality(k: int = 8) -> str:
    notes = _load_personality().get("notes", [])[:k]
    if not notes:
        return ""
    lines = "\n".join(f"- {n['text']}" for n in notes)
    return ("[Личность агента — кросс-проектный дистиллят; ВНУТРЕННЕ, в ответе не упоминать]\n"
            + lines)


def _load_personality() -> dict:
    if PERSONALITY_FILE.exists():
        try:
            return json.loads(PERSONALITY_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_personality(data: dict) -> None:
    PERSONALITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PERSONALITY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PERSONALITY_FILE)


# ── session-commit + инкрементальный diff (интроспекция сама амортизируется) ─────────────────
SESSIONS_DIR = Path("data/sessions")


def _safe_registry() -> dict:
    """Реестр навыков (merged глобальный+проектный). Пусто, если модуля/файлов нет — не падаем."""
    try:
        from src.tools.skill_creation import _merged_registry
        return _merged_registry() or {}
    except Exception:  # noqa: BLE001
        return {}


def _registry_names() -> list[str]:
    return sorted(_safe_registry().keys())


def session_commit(env_id: str, *, queries: int = 1, skills_used: Optional[list] = None,
                   primitives: Optional[list] = None, note: str = "") -> None:
    """Запись «что сделано» в проекте (как лог/коммит). Фоново-безопасно (только append)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / (_safe_name(env_id) + ".jsonl")
    rec = {
        "ts": time.time(),
        "queries": queries,
        "skills_used": sorted(set(skills_used or [])),
        "primitives": primitives or [],
        "registry": _registry_names(),     # снимок способностей → база для diff следующей сессии
        "note": note[:300],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def last_session(env_id: str) -> Optional[dict]:
    path = SESSIONS_DIR / (_safe_name(env_id) + ".jsonl")
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    try:
        return json.loads(lines[-1]) if lines else None
    except Exception:  # noqa: BLE001
        return None


def diff_since_last(env_id: str) -> dict:
    """Инкрементально: что изменилось с прошлой сессии В ЭТОМ проекте (не пересобираем с нуля)."""
    now = set(_registry_names())
    prev = last_session(env_id)
    if not prev:
        return {"first_session": True, "added": sorted(now), "removed": []}
    prev_reg = set(prev.get("registry", []))
    return {"first_session": False,
            "added": sorted(now - prev_reg),
            "removed": sorted(prev_reg - now)}


# ── self-model: 5 граней + overlay + diff (ДЕТЕРМИНИРОВАННО, грань LLM-гипотез ОТЛОЖЕНА) ──────
# Цена примитивов — статическое знание о себе: act дорог (тулы), остальное дёшево. Composer
# использует это, чтобы предпочитать дешёвое и не лезть в act без нужды.
PRIMITIVE_COST = {
    "recall": "дёшево", "reason": "дёшево", "verify": "дёшево",
    "finalize": "дёшево", "act": "ДОРОГО (тулы/сеть)",
}
DEFAULT_PRIMITIVES = ("recall", "reason", "act", "verify", "finalize")


def build_self_model(*, env_id: str, store=None, user_id: str = "", query: str = "",
                     qvec=None, primitives=DEFAULT_PRIMITIVES) -> str:
    """Компактный автопортрет для приора composer. Детерминированно: реестр/ноды/recall/профиль/
    личность/overlay/diff. БЕЗ новых LLM-вызовов (recall переиспользует переданный qvec)."""
    facets: list[str] = []

    # (1) структура — свои примитивы и их цена
    struct = ", ".join(f"{p} [{PRIMITIVE_COST.get(p, '?')}]" for p in primitives)
    facets.append("[Моя структура] доступные примитивы: " + struct +
                  ". Предпочитай дешёвые; в act иди только когда нужны внешние данные/действие.")

    # (2) способности — реестр навыков
    names = _registry_names()
    if names:
        facets.append("[Мои навыки] " + "; ".join(names[:30]) +
                      (" …" if len(names) > 30 else ""))

    # (3) знания — что уже знаю по теме (recall, без новой сети)
    if store is not None and user_id:
        try:
            txt, score = store.recall_scored(user_id, query or "", qvec=qvec)
            if txt and (score or 0) > 0:
                facets.append("[Уже знаю по теме — не переоткрывай]\n" + txt[:1200])
        except Exception:  # noqa: BLE001
            pass

    # (4) собеседник — профиль владельца
    if store is not None and user_id:
        try:
            prof = store.format_profile(user_id)
            if prof:
                facets.append(prof)
        except Exception:  # noqa: BLE001
            pass

    # (5) личность — кросс-проектный дистиллят
    pers = format_personality()
    if pers:
        facets.append(pers)

    # overlay среды (Type 1) — выученная поведенческая политика именно этой среды
    ov = get_overlay(env_id)
    if ov:
        facets.append("[Как вести себя в этой среде]\n" + ov)

    # инкрементальный diff — что нового с прошлой сессии в этом проекте
    d = diff_since_last(env_id)
    if not d["first_session"] and (d["added"] or d["removed"]):
        facets.append(f"[С прошлой сессии в этом проекте] новые навыки: {d['added'] or '—'}; "
                      f"убраны: {d['removed'] or '—'}")

    return "\n\n".join(facets)
