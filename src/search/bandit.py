"""
Per-user бандит-прайор выбора режима мышления (контур C, develop.md).

Few-shots маршрутизации харвестятся только из УСПЕХОВ; негативное свидетельство («у этого
юзера задачи такого типа в режиме fast регулярно проваливаются») терялось. Здесь оно
учитывается: по похожим эпизодам юзера (любой исход) строится Beta-постериор на режим,
Thompson-сэмплинг даёт рекомендацию.

Решения по рамкам CLAUDE.md:
- ядро не трогается: прайор подмешивается в СУЩЕСТВУЮЩИЙ слот memory_context reflexion;
- прайор, НЕ диктат: режим по-прежнему выбирает Self-Reflexion Choice по содержанию задачи
  (бандит крутит только свидетельства выбора, не промпты);
- ноль LLM-вызовов; пустая история → '' → поведение без изменений (деградация в текущее);
- контекст бандита = кластер похожих запросов (Jaccard, как habits/feedback) — это
  contextual bandit без отдельной модели признаков.
"""
from __future__ import annotations

import random
from typing import Optional

from src.memory.store import MemoryStore

# Минимум наблюдений по кластеру, чтобы прайор вообще показывать (1-2 эпизода — шум).
MIN_EVIDENCE = 3


def mode_prior(store: MemoryStore, user_id: str, query: str,
               rng: Optional[random.Random] = None) -> str:
    """Строка-прайор для reflexion ('' если свидетельств мало). rng — для теста."""
    rows = store.similar_episodes(user_id, query)
    if len(rows) < MIN_EVIDENCE:
        return ""
    stats: dict[str, list[int]] = {}  # mode -> [succ, fail]
    for r in rows:
        mode = r["mode"] or "deliberate"
        ok = r["outcome"] == "ok"
        s = stats.setdefault(mode, [0, 0])
        s[0 if ok else 1] += 1
    rng = rng or random.Random()
    # Thompson: sample ~ Beta(1+succ, 1+fail) на режим; рекомендация = максимум сэмпла.
    best_mode, best_sample = "", -1.0
    parts = []
    for mode, (succ, fail) in sorted(stats.items()):
        sample = rng.betavariate(1 + succ, 1 + fail)
        parts.append(f"{mode} — {succ} успех(ов)/{fail} неудач(и)")
        if sample > best_sample:
            best_mode, best_sample = mode, sample
    return (f"[Прайор выбора режима — опыт ЭТОГО пользователя на похожих задачах: "
            f"{'; '.join(parts)}. Thompson-сэмплинг предлагает «{best_mode}». "
            f"Это ПРАЙОР, не приказ: решай по содержанию задачи, но учитывай, какие режимы "
            f"у этого пользователя реально срабатывали, а какие проваливались.]")
