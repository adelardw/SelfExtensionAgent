"""
Единая точка для LLM-клиентов через OpenRouter — чтобы base_url/ключ не дублировались
и нигде не были забыты.
"""
from __future__ import annotations

import os

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def api_key() -> str | None:
    return os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def chat(model: str, temperature: float = 0.0):
    """ChatOpenAI, привязанный к OpenRouter (общий base_url и ключ)."""
    from langchain_openai.chat_models import ChatOpenAI

    return ChatOpenAI(api_key=api_key(), base_url=OPENROUTER_BASE, model=model, temperature=temperature)
