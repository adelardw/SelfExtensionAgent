"""
Самодиагностика: агент сам находит свои «косяки» из трейсов и статистики качества.

Детекция — на статистике (без LLM, дёшево): медленные ноды, ошибки, «залипания»
(retry storms), деградация уверенности. Отчёт можно отдать в self-learning как
сигнал, ГДЕ чинить, либо показать пользователю/в лог.
"""
from __future__ import annotations

from .tracer import trace_store


def diagnose(memory_store=None, user_id: str = "default", since_hours: float = 24.0) -> dict:
    findings: list[str] = []

    stats = trace_store.node_stats(since_hours)
    for s in stats:
        if s["errors"] and s["calls"]:
            rate = s["errors"] / s["calls"]
            if rate >= 0.2:
                findings.append(f"Нода '{s['node']}' падает в {rate:.0%} вызовов ({s['errors']}/{s['calls']}).")
        if s["avg_ms"] and s["avg_ms"] > 8000:
            findings.append(f"Нода '{s['node']}' медленная: ~{s['avg_ms']/1000:.1f}с в среднем.")

    storms = trace_store.retry_storms(since_hours)
    if storms:
        findings.append(
            f"Залипание шагов в {len(storms)} проходах (макс {storms[0]['step_calls']} ретраев одного шага) "
            f"— возможно, плохой done_check или невыполнимый подшаг."
        )

    mode_distribution: dict = {}
    if memory_store is not None:
        trend = memory_store.quality_trend(user_id)
        if trend["trend"] == "declining":
            findings.append(
                f"Деградация качества: уверенность {trend['prev_avg']} → {trend['recent_avg']} "
                f"(по {trend['n']} валидированным ответам)."
            )
        # Бюджет-метрика: какой % запросов реально уходит в дешёвый fast-путь.
        mode_distribution = memory_store.mode_stats(user_id)
        if mode_distribution["total"] >= 20 and mode_distribution["cheap_share"] < 0.2:
            findings.append(
                f"Лишь {mode_distribution['cheap_share']:.0%} запросов идут дешёвым путём (fast/clarify) "
                f"при {mode_distribution['total']} эпизодах — проверь пороги reflexion-роутера (латентность/бюджет)."
            )

    # Тихие деградации (broad-except fallback'и) этого процесса — делаем НАБЛЮДАЕМЫМИ (долг #5):
    # частые reflexion_failed/decompose_failed/step_validation_skipped = системный сбой, агент
    # молча тупеет (напр. кончился ключ эмбеддера), а не падает.
    from src.runtime import degradation
    degr = degradation.snapshot()
    if degr:
        findings.append(f"Тихие деградации (fallback'и) за процесс: {degr} "
                        f"— суммарно {sum(degr.values())} (систем-здоровье, кумулятивно).")

    return {
        "findings": findings,
        "node_stats": [dict(s) for s in stats],
        "mode_distribution": mode_distribution,
        "degradations": degr,
        "healthy": len(findings) == 0,
    }
