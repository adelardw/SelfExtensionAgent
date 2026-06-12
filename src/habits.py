"""
Привычки: само-расширение из ПОВТОРОВ (контур B, develop.md).

Само-расширение реактивно («нет способности → создать»); парадигма требует большего:
повторяющиеся ДОРОГИЕ deliberate-прогоны юзера конвертировать в дешёвые персональные
навыки («способность=цель, бюджет=констрейнт»; «универсальность = сумма персонализаций»).

Механика — через уже существующие каналы, без новых LLM-путей и без правки графа:
  reflect (после успешного дорогого прогона) → k похожих успехов в эпизодах =
  ПРИВЫЧКА → факт-директива «привычка: …» в памяти → recall инъектирует его в
  memory_context → router при следующем таком запросе выбирает create_skill →
  существующая ветка создаёт навык (SGR-ревью + smoke) → reflect помечает привычку
  закрытой (✅) → дальше задача идёт быстрым путём через skill_selector.

Детекция и директива — ноль LLM-вызовов; платный шаг (создание навыка) происходит
только в уже существующей ветке и под её защитами. Факты видны в /facts — обратимо.
"""
from __future__ import annotations

from typing import Optional

from .memory.store import MemoryStore, _overlap

_PREFIX = "привычка: "
# Порог похожести кластера (Jaccard по токенам; 0.35 в feedback.py считается переформулировкой).
HABIT_SIM = 0.4


def _existing(store: MemoryStore, user_id: str, query: str, min_sim: float):
    """Факт-привычка (включая закрытые ✅ — они глушат повторное флагование навсегда)."""
    for f in store.get_facts(user_id):
        if f["key"].startswith(_PREFIX) and _overlap(query, f["key"][len(_PREFIX):]) >= min_sim:
            return f
    return None


def maybe_flag(store: MemoryStore, user_id: str, query: str, k: int = 3,
               min_sim: float = HABIT_SIM) -> Optional[str]:
    """
    Звать из reflect ПОСЛЕ записи эпизода успешного deliberate/heavy-прогона (текущий
    эпизод уже в базе и входит в счёт k). Возвращает ключ записанной директивы или None.
    """
    if _existing(store, user_id, query, min_sim):
        return None
    similars = store.similar_successes(user_id, query, min_sim=min_sim)
    if len(similars) < k:
        return None
    key = _PREFIX + query[:60]
    store.add_fact(
        user_id=user_id, key=key,
        value=(f"Задача такого типа решалась уже {len(similars)} раз(а), каждый раз дорого "
               "(многошагово). При СЛЕДУЮЩЕМ таком запросе выбери маршрут create_skill: "
               "создай переиспользуемый навык, автоматизирующий её."),
        importance=0.75, tags=["habit", "self-extension"],
    )
    return key


def resolve(store: MemoryStore, user_id: str, query: str, skill_name: str,
            min_sim: float = HABIT_SIM) -> bool:
    """Навык для привычки создан → закрыть директиву (иначе она вечно требовала бы create_skill)."""
    f = _existing(store, user_id, query, min_sim)
    if not f or str(f["value"]).startswith("✅"):
        return False
    store.add_fact(
        user_id=user_id, key=f["key"],
        value=f"✅ навык '{skill_name}' уже создан для этой задачи — НЕ создавай заново, используй его.",
        importance=0.6, tags=["habit", "resolved"],
    )
    return True
