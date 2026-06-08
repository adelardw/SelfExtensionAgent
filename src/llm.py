"""
Единая точка для LLM-клиентов. Провайдер переключается в config.yml: `provider`.
  • openrouter — облако (OpenRouter), ключ OPEN_ROUTER_API_KEY.
  • ollama     — локально (http://localhost:11434/v1), бесплатно по токенам.

Так base_url/ключ/выбор модели не дублируются и переключаются одной строкой.
"""
from __future__ import annotations

import os

from omegaconf import OmegaConf

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_cfg = OmegaConf.load("config.yml")


def provider() -> str:
    return _cfg.get("provider", "openrouter")


def api_key() -> str | None:
    return os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def _is_code_model(model: str) -> bool:
    return model == _cfg.get("code_model", {}).get("name")


def resolve_model(model: str) -> tuple[str, str, str]:
    """
    По роли/имени модели и провайдеру → (base_url, api_key, фактическое_имя_модели).
    Для ollama имя облачной модели заменяется на локальную (code → code_model, иначе model).
    """
    if provider() == "ollama":
        oll = _cfg.get("ollama", {})
        base = oll.get("base_url", "http://localhost:11434/v1")
        name = oll.get("code_model") if _is_code_model(model) else oll.get("model")
        return base, "ollama", name or "llama3.1"
    return OPENROUTER_BASE, (api_key() or ""), model


def chat(model: str, temperature: float = 0.0):
    """ChatOpenAI, привязанный к текущему провайдеру (OpenRouter/Ollama)."""
    from langchain_openai.chat_models import ChatOpenAI

    base, key, name = resolve_model(model)
    return ChatOpenAI(api_key=key or "x", base_url=base, model=name, temperature=temperature)
