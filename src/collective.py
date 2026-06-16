"""
Коллективный ярус рецептов (контур G) — «рекомендательная система поведения».

Идея юзера: похожим людям — примерно одно и то же поведение агента. Рецепт, ДОКАЗАВШИЙ
себя у конкретного юзера (winrate-гейт), промоутится в общий пул инсталляции вместе с
отпечатком профиля источника («каким людям это помогало»). Новый юзер без собственного
рецепта получает коллективный — если похож запрос И профили пересекаются (item-based
рекомендация: item=рецепт, фичи юзера=роли профиля). Универсальность = сумма
персонализаций: лучшие персонализации становятся best-practice для похожих.

Защиты (та же дисциплина, что у глобальных few-shots):
- промоутится ТОЛЬКО проверенное (uses ≥ 2, winrate ≥ 0.66) — один юзер не отравит всех;
- запросы с сигнатурами инъекций НЕ промоутятся (не учиться на взломе);
- личный рецепт ВСЕГДА приоритетнее коллективного (персонализация > усреднение);
- глобальные рецепты продолжают копить win/lose у новых юзеров — не сработавшая у
  «похожих» рекомендация самоудаляется (дрейф тонет);
- пул живёт в ЭТОЙ инсталляции (SQLite), наружу ничего не уходит.
"""
from __future__ import annotations

import json
from typing import Optional

from .improve.safety import is_unsafe_to_learn
from .memory.store import MemoryStore, _overlap

GLOBAL_UID = "__global__"
PROMOTE_USES = 2          # минимум применений у источника
PROMOTE_WINRATE = 0.66    # минимум доля побед
PROFILE_GATE = 0.08       # минимальное пересечение профилей (Jaccard), когда оба известны


def profile_text(store: MemoryStore, user_id: str) -> str:
    """Отпечаток профиля: ЗНАЧЕНИЯ роль-фактов одной строкой (Jaccard-матчинг). Ключи
    не включаем — служебные слова («роль») есть у всех и дают ложное пересечение."""
    try:
        return " ".join(str(f["value"]) for f in store.get_role_facts(user_id))
    except Exception:  # noqa: BLE001
        return ""


def _redact_struct(obj):
    """Рекурсивно редактирует PII в строковых листьях JSON-структуры (план = list[dict]).
    Числовые литералы не трогаем — сохраняем валидность структуры (а не сериализованной строки)."""
    from .improve.safety import redact_pii
    if isinstance(obj, str):
        return redact_pii(obj)[0]
    if isinstance(obj, list):
        return [_redact_struct(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _redact_struct(v) for k, v in obj.items()}
    return obj


def maybe_promote(store: MemoryStore, user_id: str, recipe_id: int) -> bool:
    """Проверенный личный рецепт → общий пул (с отпечатком профиля источника)."""
    r = store.get_recipe(recipe_id)
    if not r or r["user_id"] == GLOBAL_UID:
        return False
    uses = r["uses"] or 0
    if uses < PROMOTE_USES or (r["wins"] or 0) / uses < PROMOTE_WINRATE:
        return False
    if is_unsafe_to_learn(r["query"]):
        return False  # запрет на обучение по взлому — и коллективно тоже
    if store.find_recipe(GLOBAL_UID, r["query"], min_sim=0.45):
        return False  # похожий best-practice уже есть — не дублируем (кластер пула шире)
    # Анти-PII (Thread 2c, SEC-4): к ДРУГИМ юзерам пересекают границу не только query, но и PLAN
    # (goal/done_check шагов эхо-несут специфику задачи) и PROFILE (значения роль-фактов источника).
    # Редактируем перс-данные (email/телефон/карта) во ВСЁМ, что шарится. Похожесть-матчинг и
    # форма плана живут на оставшихся токенах (структура JSON сохраняется — редакция точечная).
    from .improve.safety import redact_pii
    safe_query, _n = redact_pii(r["query"])
    safe_plan = _redact_struct(json.loads(r["plan"] or "[]"))   # редакция строк-листьев плана
    safe_profile, _nf = redact_pii(profile_text(store, user_id))
    store.add_recipe(GLOBAL_UID, safe_query, json.loads(r["skills"]),
                     safe_plan, r["mode"] or "deliberate",
                     profile=safe_profile)
    return True


def find_recipe(store: MemoryStore, user_id: str, query: str,
                min_sim: float = 0.55) -> Optional[object]:
    """ЛИЧНЫЙ рецепт → иначе КОЛЛЕКТИВНЫЙ (по похожести запроса + профиль-гейт).
    Профиль-гейт мягкий: пустой профиль (новый юзер / рецепт без отпечатка) пропускается —
    рекомендация по запросу; известные НЕпересекающиеся профили — отсечка (чужой тип задач)."""
    own = store.find_recipe(user_id, query, min_sim=min_sim)
    if own:
        return own
    g = store.find_recipe(GLOBAL_UID, query, min_sim=min_sim)
    if not g:
        return None
    rp, up = (g["profile"] or "").strip(), profile_text(store, user_id).strip()
    if rp and up and _overlap(rp, up) < PROFILE_GATE:
        return None
    return g
