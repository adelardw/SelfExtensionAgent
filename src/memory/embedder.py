"""
Embedder — точка расширения для семантического поиска по памяти.

По умолчанию память работает БЕЗ эмбеддингов (NullEmbedder): релевантность
считается по пересечению токенов. Это нулевая стоимость и нулевые зависимости.

Когда нужен качественный семантический recall — включи embeddings в config.yml.
OpenAIEmbedder использует уже установленный openai-клиент. Для приватного
on-device варианта здесь же легко добавить OllamaEmbedder (тот же интерфейс).
"""
from __future__ import annotations

import math
import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Интерфейс эмбеддера. enabled=False → память падает на keyword-поиск."""

    enabled: bool

    def embed(self, text: str) -> Optional[list[float]]:
        ...


class NullEmbedder:
    """Заглушка: эмбеддинги выключены, релевантность считается по токенам."""

    enabled = False

    def embed(self, text: str) -> Optional[list[float]]:
        return None

    async def aembed(self, text: str) -> Optional[list[float]]:
        return None


class OpenAIEmbedder:
    """
    Эмбеддер через OpenAI-совместимый API. Приоритет — OpenRouter (тот же ключ
    OPEN_ROUTER_API_KEY, отдельный OPENAI_API_KEY не нужен), затем OpenAI.
    Включается только если найден ключ — иначе ведёт себя как NullEmbedder.
    """

    _OPENROUTER = "https://openrouter.ai/api/v1"

    def __init__(self, model: Optional[str] = None):
        # Ollama-провайдер: локальные эмбеддинги.
        try:
            from omegaconf import OmegaConf

            _c = OmegaConf.load("config.yml")
            if _c.get("provider") == "ollama":
                oll = _c.get("ollama", {})
                self._key, self._base = "ollama", oll.get("base_url", "http://localhost:11434/v1")
                self.model = model or oll.get("embed_model", "nomic-embed-text")
                self.enabled = True
                self._client = None
                try:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=self._key, base_url=self._base,
                                          timeout=10, max_retries=0)
                except Exception:  # noqa: BLE001
                    self.enabled = False
                return
        except Exception:  # noqa: BLE001
            pass

        or_key = os.getenv("OPEN_ROUTER_API_KEY")
        oa_key = os.getenv("OPENAI_API_KEY")
        if or_key:
            self._key, self._base = or_key, self._OPENROUTER
            self.model = model or "openai/text-embedding-3-small"
        elif oa_key:
            self._key, self._base = oa_key, None  # дефолтный OpenAI base_url
            self.model = model or "text-embedding-3-small"
        else:
            self._key, self._base, self.model = None, None, (model or "text-embedding-3-small")

        self.enabled = bool(self._key)
        self._client = None
        if self.enabled:
            try:
                from openai import OpenAI

                # таймаут+0 ретраев: sync embed-вызов блокирует event loop, без потолка
                # он морозил весь прогон на дефолтные 600с (видно в eval как «зомби»).
                _kw = dict(timeout=10, max_retries=0)
                self._client = (
                    OpenAI(api_key=self._key, base_url=self._base, **_kw) if self._base
                    else OpenAI(api_key=self._key, **_kw)
                )
            except Exception as e:  # noqa: BLE001
                print(f"[Embedder] init failed, fallback to keyword search: {e}")
                self.enabled = False

    def embed(self, text: str) -> Optional[list[float]]:
        if not self.enabled or not text.strip():
            return None
        try:
            resp = self._client.embeddings.create(model=self.model, input=text[:8000])
            return resp.data[0].embedding
        except Exception as e:  # noqa: BLE001
            print(f"[Embedder] embed failed: {e}")
            return None

    async def aembed(self, text: str) -> Optional[list[float]]:
        """Async-обёртка: sync-вызов в to_thread, чтобы не блокировать event loop."""
        import asyncio
        return await asyncio.to_thread(self.embed, text)


def cosine(a: list[float], b: list[float]) -> float:
    """Косинусная близость двух векторов. 0.0 при вырожденных входах."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_embedder(use_embeddings: bool, model: Optional[str] = None) -> Embedder:
    """Фабрика по флагу из конфига. model — опц. имя модели эмбеддингов (OpenRouter/OpenAI)."""
    if use_embeddings:
        emb = OpenAIEmbedder(model=model)
        if emb.enabled:
            return emb
        print("[Embedder] embeddings=true, но ключ не найден → keyword-поиск.")
    return NullEmbedder()
