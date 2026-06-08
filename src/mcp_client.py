"""
Автоподключение к MCP-серверам и использование их инструментов.

Security-баланс: автоматически подключаемся ТОЛЬКО к доверенным серверам из allowlist
(TRUSTED_SERVERS). Произвольный MCP из интернета — через human-gate (propose_untrusted),
т.к. это запуск чужого кода с нашими доступами.

CATALOG — каталог известных надёжных серверов, из которого агент может «сам найти»
нужный под задачу (по ключевым словам). Транспорт — stdio через uvx (Python-серверы).
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Optional

# Официальный реестр MCP — настоящий discovery «найти сервер под задачу».
REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"

# Доверенные серверы: автоподключение разрешено.
TRUSTED_SERVERS: dict[str, dict] = {
    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "transport": "stdio"},
}

# Каталог известных надёжных MCP (official/Python, без ключей) — для «сам найди под задачу».
CATALOG: dict[str, dict] = {
    "fetch": {"spec": TRUSTED_SERVERS["fetch"], "keywords": ["url", "сайт", "страниц", "fetch", "веб", "web", "новост"],
              "desc": "Загрузка и markdown-извлечение веб-страниц"},
    "time": {"spec": {"command": "uvx", "args": ["mcp-server-time"], "transport": "stdio"},
             "keywords": ["врем", "time", "часов", "timezone", "дата", "который час"], "desc": "Текущее время и таймзоны"},
}


def suggest_server(query: str) -> Optional[str]:
    """Подобрать сервер из каталога под запрос по ключевым словам (агент «находит» MCP)."""
    q = query.lower()
    best, score = None, 0
    for name, meta in CATALOG.items():
        s = sum(1 for kw in meta["keywords"] if kw in q)
        if s > score:
            best, score = name, s
    return best


_tools_cache: dict[frozenset, list] = {}


async def get_mcp_tools(servers: Optional[list[str]] = None) -> list:
    """
    Подключается к доверенным серверам (stdio) и возвращает их инструменты как
    LangChain-tools. Недоверенные — игнорируются. Кэширует по набору серверов,
    чтобы не пере-спавнивать сервер на каждом шаге цикла.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    names = servers or list(TRUSTED_SERVERS)
    cfg = {n: TRUSTED_SERVERS[n] for n in names if n in TRUSTED_SERVERS}
    if not cfg:
        return []
    key = frozenset(cfg)
    if key in _tools_cache:
        return _tools_cache[key]
    try:
        client = MultiServerMCPClient(cfg)
        tools = await client.get_tools()
        _tools_cache[key] = tools
        return tools
    except Exception as e:  # noqa: BLE001
        print(f"[mcp] connect failed: {e}")
        return []


def discover_mcp(query: str, limit: int = 25) -> list[dict]:
    """
    Настоящий discovery: ищет MCP-серверы в ОФИЦИАЛЬНОМ реестре под задачу.
    Реестр ищет по короткому ключевому слову, поэтому пробуем полный запрос, затем
    отдельные значимые слова (Latin-токены вроде 'sqlite','fetch','github' работают
    даже в русских запросах). Возвращает запускаемых кандидатов (uvx pypi / http без ключа) —
    это ПРЕДЛОЖЕНИЯ (untrusted): подключение только после approve_server (human-gate).
    """
    terms = [query] + sorted(set(re.findall(r"[A-Za-z]{4,}", query)), key=len, reverse=True)
    for term in terms:
        res = _search_registry(term, limit)
        if res:
            return res
    return []


def _search_registry(term: str, limit: int) -> list[dict]:
    url = f"{REGISTRY_URL}?{urllib.parse.urlencode({'search': term, 'limit': limit})}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "self-extension-agent"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8", "ignore"))
    except Exception as e:  # noqa: BLE001
        print(f"[mcp] discovery failed: {e}")
        return []

    out, seen = [], set()
    for e in data.get("servers", []):
        s = e.get("server", {})
        name = s.get("name", "")
        if name in seen:
            continue
        spec = kind = pkg = None
        # 1) локальный запуск через uvx (pypi) — предпочтительно
        for p in s.get("packages", []):
            if (p.get("registryType") or "").lower() == "pypi" and p.get("identifier"):
                spec = {"command": "uvx", "args": [p["identifier"]], "transport": "stdio"}
                kind, pkg = "pypi", p["identifier"]
                break
        # 2) удалённый http/sse БЕЗ обязательного ключа
        if not spec:
            for r in s.get("remotes", []):
                rtype, url = r.get("type", ""), r.get("url")
                needs_key = any("{" in (h.get("value") or "") for h in r.get("headers", []))
                if url and not needs_key and rtype in ("streamable-http", "streamable_http", "sse"):
                    transport = "sse" if rtype == "sse" else "streamable_http"
                    spec = {"transport": transport, "url": url}
                    kind, pkg = "remote", url
                    break
        if spec:
            seen.add(name)
            out.append({
                "name": name,
                "description": (s.get("description") or "")[:160],
                "package": pkg,
                "spec": spec,
                "source": f"mcp-registry/{kind}",
            })
    return out


def approve_server(name: str, spec: dict) -> str:
    """Human-gate: явно одобрить найденный сервер → добавить в trusted (станет авто-подключаемым)."""
    TRUSTED_SERVERS[name] = spec
    return name


def propose_untrusted(name: str, spec: dict) -> dict:
    """
    Человеческий гейт для НЕдоверенного MCP: не подключаем, а возвращаем предложение
    на подтверждение. Подключение чужого кода без approval запрещено.
    """
    return {"status": "needs_approval", "name": name, "spec": spec,
            "note": "Подключение произвольного MCP — выполнение чужого кода. Нужно явное подтверждение пользователя."}
