"""
Единая точка для LLM-клиентов. Провайдер в config.yml: `provider` (openrouter | ollama).
Модели выбираются по РОЛИ (fast | code | embed), а не по имени — без догадок и костылей.
Фолбэк: provider=ollama → используем Ollama, если он реально генерирует, иначе OpenRouter.
"""
from __future__ import annotations

import json
import os
import urllib.request

from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()  # ключи доступны и при прямом использовании (media.py и т.п.), не только через agent

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_cfg = OmegaConf.load("config.yml")
_active: str | None = None  # кэш активного провайдера за сессию
_override: dict = {"provider": None, "model": None}  # рантайм-выбор из CLI (/model)


def set_provider(provider_name: str | None, model: str | None = None) -> None:
    """Рантайм-переключение провайдера/модели (CLI /model). Сбрасывает кэш health-check."""
    global _active
    _override["provider"] = provider_name
    _override["model"] = model
    _active = None


def api_key() -> str | None:
    return os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def _ollama_works() -> bool:
    """Ollama не просто отвечает, а РЕАЛЬНО генерирует (ловит сломанный движок)."""
    oll = _cfg.get("ollama", {})
    base = oll.get("base_url", "http://localhost:11434/v1").rstrip("/").removesuffix("/v1")
    model = oll.get("model", "llama3.1")
    try:
        req = urllib.request.Request(
            base + "/api/generate",
            data=json.dumps({"model": model, "prompt": "hi", "stream": False,
                             "options": {"num_predict": 1}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            json.loads(r.read())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[llm] ⚠ Ollama не генерирует ({str(e)[:80]}) → fallback на OpenRouter.")
        return False


def provider() -> str:
    global _active
    if _active is not None:
        return _active
    p = _override["provider"] or _cfg.get("provider", "openrouter")
    if p == "ollama" and not _ollama_works():
        p = "openrouter"
    _active = p
    return p


def model_for(role: str) -> str:
    """Имя модели для роли при активном провайдере. role: fast | code | deep | embed.
    'deep' — редкие тяжёлые вызовы (heavy-ревью); фолбэк на code-модель."""
    if provider() == "ollama":
        oll = _cfg.get("ollama", {})
        if _override["model"] and role in ("fast", "code", "deep"):
            return _override["model"]  # модель, выбранная из CLI
        fast = oll.get("model", "llama3.1")
        return {"fast": fast, "code": oll.get("code_model") or fast,
                "deep": oll.get("deep_model") or oll.get("code_model") or fast,
                "embed": oll.get("embed_model", "nomic-embed-text")}.get(role, fast)
    if role == "code":
        return _cfg.get("code_model", {}).get("name", "gpt-4o-mini")
    if role == "deep":
        return (_cfg.get("deep_model", {}) or {}).get("name") or _cfg.get("code_model", {}).get("name", "gpt-4o-mini")
    if role == "embed":
        return _cfg.get("memory", {}).get("embedding_model", "openai/text-embedding-3-small")
    return _cfg.get("model", {}).get("name", "gpt-4o-mini")


def _base_and_key() -> tuple[str, str]:
    if provider() == "ollama":
        return _cfg.get("ollama", {}).get("base_url", "http://localhost:11434/v1"), "ollama"
    return OPENROUTER_BASE, (api_key() or "")


def chat(role: str = "fast", temperature: float = 0.0):
    """ChatOpenAI для активного провайдера + модель по роли (fast|code|deep).
    К каждому клиенту привязан run-budget callback — все вызовы (включая под-агентов)
    учитываются в токен-бюджете прогона (см. runbudget)."""
    from langchain_openai.chat_models import ChatOpenAI

    from .runbudget import callback

    base, key = _base_and_key()
    # ЖЁСТКИЙ ТАЙМАУТ + мало ретраев: зависший API-вызов не должен морозить агента на
    # дефолтные 600с × ретраи (это плодило «зомби на 0% CPU» в eval и фризы в проде).
    return ChatOpenAI(api_key=key or "x", base_url=base, model=model_for(role),
                      temperature=temperature, callbacks=[callback()],
                      timeout=90, max_retries=1)


def active_summary() -> str:
    """Строка для баннера — РЕАЛЬНО активные провайдер и модели."""
    p = provider()
    fast, code = model_for("fast"), model_for("code")
    return f"{p} · {fast}" + (f" · код: {code}" if code != fast else "")
