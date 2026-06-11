"""Само-расширение к data-MCP: ОБЩИЙ фильтр релевантности (не подключать мусор). Офлайн."""
import asyncio
import re

import src.mcp_client as MC
from src.mcp_client import _relevance


def test_relevance_scores_by_overlap():
    terms = set(re.findall(r"[a-z]{3,}", "movies film rating netflix"))
    movie = {"name": "pipeworx/movies", "description": "Movie database with ratings and runtimes"}
    sales = {"name": "servicepal/mcp", "description": "Calculate missed calls and revenue loss for sales"}
    assert _relevance(movie, terms) > _relevance(sales, terms)


def test_connect_skips_irrelevant_and_dead(monkeypatch):
    # discover вернул мусор (sales) + релевантный movie; movie живой → подключаемся к movie
    cands = [
        {"name": "servicepal", "description": "sales revenue calls", "spec": {"transport": "streamable_http", "url": "x"}},
        {"name": "movies-db", "description": "movie film rating database", "spec": {"transport": "streamable_http", "url": "y"}},
    ]
    monkeypatch.setattr(MC, "discover_mcp", lambda q, n=8: cands)

    async def _fake_domain(q):
        return "movies"

    monkeypatch.setattr(MC, "_extract_domain", _fake_domain)

    class _T:
        def __init__(self, name):
            self.name = name

    async def _fake_get(servers):
        # подключается ТОЛЬКО релевантный (movies-db); проверяем, что мусор не выбран первым
        return [_T("search_movies")] if servers == ["movies-db"] else []

    monkeypatch.setattr(MC, "get_mcp_tools", _fake_get)

    name, tools = asyncio.run(MC.try_connect_discovered("highest rated movie", max_try=3))
    assert name == "movies-db" and len(tools) == 1


def test_connect_returns_none_when_no_candidates(monkeypatch):
    monkeypatch.setattr(MC, "discover_mcp", lambda q, n=8: [])

    async def _fake_domain(q):
        return "weather"

    monkeypatch.setattr(MC, "_extract_domain", _fake_domain)
    name, tools = asyncio.run(MC.try_connect_discovered("rain tomorrow"))
    assert name is None and tools == []
