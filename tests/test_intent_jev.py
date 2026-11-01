"""JEV-бэкенд intent-роутера (src/graph/intent.py): флаг, парсинг, кэш на запрос, прогрев из
recall, откат на kNN. Сеть замокана (httpx.post)."""
import asyncio

import httpx
import pytest

import src.llm.llm as llm
from src.graph import intent


def _resp(choice="play_media", conf=0.9):
    return {"model": "typesafe/jev-1.13", "usage": {"input_tokens": 490, "cost": 0.00002},
            "answers": {"route": {"type": "choice", "choice": choice, "confidence": conf,
                                  "probabilities": {choice: conf, "self_contained": 1 - conf}}}}


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _NoEmb:
    enabled = False


@pytest.fixture(autouse=True)
def _jev_on(monkeypatch):
    intent._JEV_CACHE.clear()
    monkeypatch.setenv("AGENT_INTENT_BACKEND", "jev")
    monkeypatch.setattr(llm, "provider", lambda: "openrouter")
    monkeypatch.setattr(llm, "api_key", lambda: "sk-test")
    monkeypatch.setattr(llm, "openrouter_base_url", lambda: "https://openrouter.ai/api/v1")
    yield
    intent._JEV_CACHE.clear()


@pytest.fixture()
def calls(monkeypatch):
    seen = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(json)
        return _Resp(_resp())
    monkeypatch.setattr(httpx, "post", fake_post)
    return seen


def test_enabled_only_for_openrouter_cloud(monkeypatch):
    assert intent.jev_enabled()
    monkeypatch.setattr(llm, "provider", lambda: "ollama")
    assert not intent.jev_enabled()
    monkeypatch.setattr(llm, "provider", lambda: "openrouter")
    monkeypatch.setattr(llm, "openrouter_base_url", lambda: "http://my-gateway:8000/v1")
    assert not intent.jev_enabled()
    monkeypatch.setattr(llm, "openrouter_base_url", lambda: "https://openrouter.ai/api/v1")
    monkeypatch.setenv("AGENT_INTENT_BACKEND", "knn")
    assert not intent.jev_enabled()


def test_parse_rejects_unknown_label():
    assert intent._jev_parse(_resp("play_media"))["label"] == "play_media"
    assert intent._jev_parse(_resp("order_pizza")) is None
    assert intent._jev_parse({}) is None


def test_request_shape_and_one_call_per_query(calls):
    r = intent.IntentRouter()
    r._embedder = _NoEmb()   # JEV работает и без эмбеддингов
    for _ in range(3):       # как _needs_web_grounding / _wants_physical_browser / _is_play_intent
        c = r.classify("  включи джаз  ")
        assert c["label"] == "play_media" and c["backend"] == "jev"
    assert len(calls) == 1
    q = calls[0]["questions"]["route"]
    assert calls[0]["state"] == "включи джаз"
    assert q["type"] == "choice" and set(q["criteria"]) == set(intent.LABELS)


def test_prefetch_warms_cache(calls):
    asyncio.run(intent.ajev_prefetch("play despacito"))
    assert len(calls) == 1
    r = intent.IntentRouter()
    r._embedder = _NoEmb()
    assert r.classify("play despacito")["label"] == "play_media"
    assert len(calls) == 1


def test_prefetch_noop_when_backend_knn(calls, monkeypatch):
    monkeypatch.setenv("AGENT_INTENT_BACKEND", "knn")
    asyncio.run(intent.ajev_prefetch("play despacito"))
    assert calls == []


def test_failure_falls_back_to_knn_and_is_cached(monkeypatch):
    n = []

    def boom(*a, **k):
        n.append(1)
        raise httpx.ConnectTimeout("timeout")
    monkeypatch.setattr(httpx, "post", boom)
    r = intent.IntentRouter()
    r._embedder = _NoEmb()
    assert r.classify("pause the music") is None   # JEV упал, kNN без эмбеддингов → None
    assert r.classify("pause the music") is None
    assert len(n) == 1                             # таймаут не ждём повторно на тот же запрос
