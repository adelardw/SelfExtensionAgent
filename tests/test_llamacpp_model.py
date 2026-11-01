"""GGUF adapter protocol, size guard and per-request reasoning, without a model download."""
import asyncio
import json

import httpx
import pytest

from src.coding.harness import Turn
from src.coding.model import LlamaCppModel, build_local_model


def mock_server(monkeypatch, *, count=4e9, content='{"action":"finish","summary":"ok"}'):
    payloads = []
    def transport(req):
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "sea-local", "meta": {"n_params": count}}]})
        if req.url.path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 16384}})
        payloads.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "timings": {"cache_n": 60, "predicted_n": 20, "predicted_ms": 500}})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(transport)))
    return payloads


def test_reasoning_and_json_schema_without_cloud(monkeypatch):
    payloads = mock_server(monkeypatch)
    model = build_local_model({"backend": "llama_cpp"})
    assert isinstance(model, LlamaCppModel)
    for effort in ("low", "high"):
        turn = asyncio.run(model([{"role": "user", "content": "fix"}], Turn, effort))
        assert turn.action == "finish"
    assert not payloads[0]["chat_template_kwargs"]["enable_thinking"]
    assert payloads[1]["chat_template_kwargs"]["enable_thinking"]
    assert payloads[0]["reasoning_effort"] == "none"
    assert payloads[0]["response_format"]["json_schema"]["schema"]["oneOf"][1]["properties"]["tool"]["enum"]
    assert model.usage()["provider"] == "llama_cpp"
    assert model.usage()["tokens"] == 240
    assert model.usage()["cached_tokens"] == 120
    assert model.usage()["generation_tokens_per_second"] == 40


@pytest.mark.parametrize("count", [None, 12e9])
def test_loaded_model_size_is_enforced(monkeypatch, count):
    payloads = mock_server(monkeypatch, count=count)
    with pytest.raises(ValueError, match="10B"):
        asyncio.run(LlamaCppModel({})([{"role": "user", "content": "hi"}], Turn, "low"))
    assert not payloads


def test_missing_final_json_is_an_error(monkeypatch):
    mock_server(monkeypatch, content="")
    with pytest.raises(ValueError, match="budget"):
        asyncio.run(LlamaCppModel({})([{"role": "user", "content": "hi"}], Turn, "high"))


def test_unknown_backend_does_not_fall_back():
    with pytest.raises(ValueError):
        build_local_model({"backend": "remote"})
