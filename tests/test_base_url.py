"""Endpoint провайдера: единый приоритет env → настройки CLI → config.yml → дефолт,
для OpenRouter-совместимого и для Ollama. Offline (без сети)."""
import src.llm.llm as L


def _no_env(monkeypatch):
    for v in ("OPENROUTER_BASE_URL", "OPENAI_BASE_URL", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(v, raising=False)


def test_openrouter_default_and_priority(monkeypatch):
    _no_env(monkeypatch)
    monkeypatch.setattr(L, "_cli_override", lambda k, d=None: None)
    monkeypatch.setattr(L, "_cfg", {})
    assert L.openrouter_base_url() == "https://openrouter.ai/api/v1"

    monkeypatch.setattr(L, "_cfg", {"base_url": "http://from-config:8000/v1"})
    assert L.openrouter_base_url() == "http://from-config:8000/v1"      # config.yml

    monkeypatch.setattr(L, "_cli_override",
                        lambda k, d=None: "http://from-cli:9000/v1" if k == "base_url" else None)
    assert L.openrouter_base_url() == "http://from-cli:9000/v1"          # настройки > config.yml

    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://from-env:7000/v1/")
    assert L.openrouter_base_url() == "http://from-env:7000/v1"          # env > всего, / срезан


def test_openai_base_url_alias(monkeypatch):
    """Стандартная переменная OpenAI-совместимых клиентов тоже принимается."""
    _no_env(monkeypatch)
    monkeypatch.setattr(L, "_cli_override", lambda k, d=None: None)
    monkeypatch.setattr(L, "_cfg", {})
    monkeypatch.setenv("OPENAI_BASE_URL", "http://compat:1234/v1")
    assert L.openrouter_base_url() == "http://compat:1234/v1"


def test_ollama_priority_and_v1_suffix(monkeypatch):
    _no_env(monkeypatch)
    monkeypatch.setattr(L, "_cli_override", lambda k, d=None: None)
    monkeypatch.setattr(L, "_cfg", {})
    assert L.ollama_base_url() == "http://localhost:11434/v1"

    monkeypatch.setattr(L, "_cfg", {"ollama": {"base_url": "http://cfg-host:11434"}})
    assert L.ollama_base_url() == "http://cfg-host:11434/v1"             # /v1 дописан

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-box:11434")
    assert L.ollama_base_url() == "http://remote-box:11434/v1"           # env > config, /v1 дописан


def test_base_and_key_uses_resolvers(monkeypatch):
    _no_env(monkeypatch)
    monkeypatch.setattr(L, "_cli_override", lambda k, d=None: None)
    monkeypatch.setattr(L, "_cfg", {})
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://box:11434")
    monkeypatch.setattr(L, "provider", lambda: "ollama")
    base, key = L._base_and_key()
    assert base == "http://box:11434/v1" and key == "ollama"             # локальному ключ не нужен

    monkeypatch.setattr(L, "provider", lambda: "openrouter")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://gw:8000/v1")
    monkeypatch.setattr(L, "api_key", lambda: "k")
    assert L._base_and_key() == ("http://gw:8000/v1", "k")


def test_base_url_source_labels(monkeypatch):
    _no_env(monkeypatch)
    monkeypatch.setattr(L, "_cli_override", lambda k, d=None: None)
    monkeypatch.setattr(L, "_cfg", {})
    monkeypatch.setattr(L, "provider", lambda: "openrouter")
    assert L.base_url_source() == "по умолчанию"
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://x/v1")
    assert "env" in L.base_url_source()
