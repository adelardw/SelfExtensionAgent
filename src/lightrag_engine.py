"""
Движок графовой БЗ на НАСТОЯЩЕМ LightRAG (пакет lightrag-hku), не самодельный граф.
Per-user инстанс: рабочая директория, LLM=наш OpenRouter (fast), эмбеддинги=OpenRouter.
Ингест строит граф сущностей+связей; запрос — гибридный (граф + вектор), multi-hop.

Best-effort: нет пакета/ключа → lightrag_available()=False, БЗ деградирует на BM25 (floor).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

_BASE = "https://openrouter.ai/api/v1"


def _key() -> Optional[str]:
    return os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def lightrag_available() -> bool:
    if not _key():
        return False
    try:
        import lightrag  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def estimate_index_cost(text: str) -> dict:
    """Прикидка цены индексации документа в граф ДО запуска (HITL: «/kb add стоит денег молча»).
    LightRAG на каждый чанк (~1200 ток., overlap 100) делает extraction + gleaning-проход,
    промпт извлечения с примерами ~3k токенов; плюс ~20% вызовов на merge-саммари сущностей.
    Токены ≈ len/3 (смешанный рус/en токенизируется хуже английского). Это ОЦЕНКА порядка, не счёт."""
    from .usage import cost_of

    tokens = max(1, len(text) // 3)
    chunks = max(1, -(-tokens // 1100))           # chunk 1200 − overlap 100
    calls = round(chunks * 2.2)                   # extraction + gleaning + ~20% merge
    usd = cost_of(calls * 4200, calls * 600)      # ~3k промпт + чанк на вход, ~600 на выход
    usd += tokens / 1e6 * 0.02                    # эмбеддинги чанков (text-embedding-3-small)
    return {"tokens": tokens, "chunks": chunks, "calls": calls, "usd": usd}


_instances: dict[str, object] = {}
_locks: dict[str, asyncio.Lock] = {}


async def _get_rag(working_dir: Path):
    """Создать/поднять кэшированный LightRAG для рабочей директории (один раз инициализируем)."""
    key = str(working_dir)
    if key in _instances:
        return _instances[key]
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        if key in _instances:
            return _instances[key]
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from .llm import model_for

        api, model = _key(), model_for("fast")
        working_dir.mkdir(parents=True, exist_ok=True)

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kw):
            kw.pop("keyword_extraction", None)  # gemini через OpenRouter не любит лишние флаги
            return await openai_complete_if_cache(
                model, prompt, system_prompt=system_prompt,
                history_messages=history_messages or [], base_url=_BASE, api_key=api, **kw)

        async def embed_func(texts):
            return await openai_embed(
                texts, model="openai/text-embedding-3-small", base_url=_BASE, api_key=api)

        rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=llm_func,
            llm_model_name=model,
            embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=embed_func),
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
        _instances[key] = rag
        return rag


async def insert(working_dir: Path, text: str) -> bool:
    """Проиндексировать документ в граф LightRAG. True — успех, False — деградация на BM25."""
    if not lightrag_available() or not text.strip():
        return False
    try:
        rag = await _get_rag(working_dir)
        await rag.ainsert(text)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[lightrag] insert failed → BM25 fallback: {type(e).__name__}: {e}")
        return False


async def query(working_dir: Path, q: str, mode: str = "hybrid",
                only_context: bool = True) -> Optional[str]:
    """Гибридный запрос к графу (граф+вектор, multi-hop). only_context=True — вернуть только
    извлечённый КОНТЕКСТ графа (без генерации ответа самим LightRAG): дешевле, агент рассуждает
    сам. None → деградация на BM25."""
    if not lightrag_available() or not (working_dir / "graph_chunk_entity_relation.graphml").exists():
        return None
    try:
        from lightrag import QueryParam
        rag = await _get_rag(working_dir)
        res = await rag.aquery(q, param=QueryParam(mode=mode, only_need_context=only_context))
        return res if isinstance(res, str) and res.strip() else None
    except Exception as e:  # noqa: BLE001
        print(f"[lightrag] query failed → BM25 fallback: {type(e).__name__}: {e}")
        return None
