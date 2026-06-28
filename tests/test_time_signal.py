"""is_time_sensitive — «сегодня» инжектится в контекст ТОЛЬКО для time-anchored запросов (поездка/
дедлайн/сезон/«сколько осталось»), а не каждый раз. Эмбеддинг-контраст без регэкспа → нужен ключ
эмбеддера (как остальные signal-тесты). Решает связность по дате без засорения контекста.
"""
import os

import pytest

needs_key = pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="сигнал на эмбеддингах — нужен ключ",
)


@needs_key
def test_time_sensitive_fires_only_on_temporal():
    from src.graph.semantic_signals import is_time_sensitive as t
    # завязано на ТЕКУЩИЙ момент → дата нужна
    assert t("спланируй поездку, еду 10 июля на 12 дней")
    assert t("сколько дней осталось до дедлайна на этой неделе")
    assert t("какое сегодня число и погода")
    # обычные запросы (в т.ч. travel без привязки к дате) → дату НЕ инжектим
    assert not t("какая столица Франции")
    assert not t("напиши код сортировки массива")
    assert not t("что посмотреть в Осаке и Киото")


@needs_key
def test_time_sensitive_accepts_precomputed_vector():
    """fires_vec: переиспользуем готовый эмбеддинг запроса (без второго embed в recall)."""
    from src.graph.semantic_signals import is_time_sensitive
    from src.memory.embedder import build_embedder

    emb = build_embedder(True)
    if not getattr(emb, "enabled", False):
        pytest.skip("эмбеддер выключен")
    qv = emb.embed("спланируй поездку, еду 10 июля")
    assert is_time_sensitive("спланируй поездку, еду 10 июля", qvec=qv) is True
