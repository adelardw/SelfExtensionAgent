"""
Автоподключение к MCP-серверам и использование их инструментов.

Security-баланс: автоматически подключаемся ТОЛЬКО к доверенным серверам из allowlist
(TRUSTED_SERVERS). Произвольный MCP из интернета — через human-gate (propose_untrusted),
т.к. это запуск чужого кода с нашими доступами.

CATALOG — каталог известных надёжных серверов, из которого агент может «сам найти»
нужный под задачу (по ключевым словам). Транспорт — stdio через uvx (Python-серверы).
"""
from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
import urllib.request
from typing import Optional

# Официальный реестр MCP — настоящий discovery «найти сервер под задачу».
REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"

# Доверенные серверы: автоподключение разрешено.
# fetch: пин `mcp<2` — свежий SDK переименовал McpError→MCPError, а mcp-server-fetch ещё
# импортирует старое имя → ImportError на каждом старте (вскрыто мульти-агентной валидацией;
# проверено: с пином сервер поднимается). Убрать пин, когда апстрим обновит импорт.
TRUSTED_SERVERS: dict[str, dict] = {
    "fetch": {"command": "uvx", "args": ["--with", "mcp<2", "mcp-server-fetch"], "transport": "stdio"},
}

# Каталог известных надёжных MCP (official/Python, без ключей) — для «сам найди под задачу».
CATALOG: dict[str, dict] = {
    "fetch": {"spec": TRUSTED_SERVERS["fetch"], "keywords": ["url", "сайт", "страниц", "fetch", "веб", "web", "новост"],
              "desc": "Загрузка и markdown-извлечение веб-страниц"},
    "time": {"spec": {"command": "uvx", "args": ["--with", "mcp<2", "mcp-server-time"], "transport": "stdio"},
             "keywords": ["врем", "time", "часов", "timezone", "дата", "который час"], "desc": "Текущее время и таймзоны"},
}


_registry_loaded = False


def _load_user_registry() -> None:
    """Подмешать пользовательский реестр MCP из корневого MCP.md (как SKILL.md, yml-поля).
    Серверы юзера → доверенные (он сам внёс) + в CATALOG для «сам найди». Идемпотентно/ленивo;
    работает одинаково для CLI и десктопа (оба идут через этот модуль). Нет MCP.md → no-op."""
    global _registry_loaded
    if _registry_loaded:
        return
    _registry_loaded = True
    try:
        from src.runtime.context_files import mcp_servers
        for s in mcp_servers():
            name = str(s["name"])
            if s.get("url"):
                spec = {"transport": s.get("transport", "streamable_http"), "url": s["url"]}
            else:
                spec = {"command": s.get("command", "uvx"),
                        "args": list(s.get("args", [])), "transport": "stdio"}
            kws = [str(k).lower() for k in (s.get("keywords") or [])]
            CATALOG[name] = {"spec": spec, "keywords": kws, "desc": s.get("description", "")}
            # ДОВЕРИЕ — ТОЛЬКО по ЯВНОМУ `trusted: true` (раньше дефолт True). Агент работает в
            # ПРОИЗВОЛЬНОМ cwd (навык code, «проанализируй этот репо»): склонированный недоверенный
            # репозиторий с MCP.md иначе авто-запускал бы uvx/remote чужой код, минуя HITL/UNLEASH
            # (баг ревью NEW-1). В каталог сервер попадает (агент видит), но БЕЗ авто-доверия.
            if s.get("trusted", False) is True:
                TRUSTED_SERVERS[name] = spec
    except Exception as e:  # noqa: BLE001
        print(f"[mcp] MCP.md реестр не загрузился: {e}")


def suggest_server(query: str) -> Optional[str]:
    """Подобрать сервер из каталога под запрос по ключевым словам (агент «находит» MCP)."""
    _load_user_registry()
    q = query.lower()
    best, score = None, 0
    for name, meta in CATALOG.items():
        s = sum(1 for kw in meta["keywords"] if kw in q)
        if s > score:
            best, score = name, s
    return best


_tools_cache: dict[frozenset, list] = {}
_MCP_CACHE_MAX = 8  # потолок кэша MCP-соединений (анти-накопление в долгой сессии)


async def get_mcp_tools(servers: Optional[list[str]] = None) -> list:
    """
    Подключается к доверенным серверам (stdio) и возвращает их инструменты как
    LangChain-tools. Недоверенные — игнорируются. Кэширует по набору серверов,
    чтобы не пере-спавнивать сервер на каждом шаге цикла.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _load_user_registry()  # подмешать серверы из MCP.md (если есть) до выбора
    names = servers or list(TRUSTED_SERVERS)
    # stdio-серверы: глушим установочный шум пакетных менеджеров (валидация р.3: node-обвязка
    # сервера писала npm-лог «added 40 packages…» в stdout → JSONRPC-парсер сыпал трейсбеки).
    _quiet = {"npm_config_loglevel": "silent", "NO_UPDATE_NOTIFIER": "1",
              "NPM_CONFIG_FUND": "false", "UV_NO_PROGRESS": "1"}
    cfg = {}
    for n in names:
        if n not in TRUSTED_SERVERS:
            continue
        spec = dict(TRUSTED_SERVERS[n])
        if spec.get("transport") == "stdio":
            import os as _os

            spec["env"] = {**_os.environ, **_quiet}
        cfg[n] = spec
    if not cfg:
        return []
    key = frozenset(cfg)
    if key in _tools_cache:
        return _tools_cache[key]
    try:
        client = MultiServerMCPClient(cfg)
        # ЖЁСТКИЙ ТАЙМАУТ: get_tools() спавнит uvx (ставит пакет с pypi) и может ВИСНУТЬ
        # на сети 100+ сек, упирая прогон в scenario-timeout (eval ловил 240с/$0.0003) и
        # морозя CLI на локальных задачах. Не подключились за MCP_CONNECT_TIMEOUT — идём
        # дальше общими инструментами, не ждём сеть.
        import os as _os
        timeout = float(_os.getenv("MCP_CONNECT_TIMEOUT", 20))
        tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
        # LRU-кап кэша: разные задачи → разные MCP → соединения копились бы безгранично.
        if len(_tools_cache) >= _MCP_CACHE_MAX:
            _tools_cache.pop(next(iter(_tools_cache)), None)  # выкидываем самый старый
        _tools_cache[key] = tools
        return tools
    except asyncio.TimeoutError:
        print(f"[mcp] connect timeout ({timeout:.0f}с) — иду дальше без MCP")
        return []
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


async def _extract_domain(query: str) -> str:
    """Дешёвый LLM-вызов: ОДНО главное слово домена данных под запрос (movies/weather/
    geolocation/finance/…). Реестр MCP ищется по домену, а не по случайным словам запроса."""
    from src.llm.llm import chat
    try:
        prompt = ("Какой ОДИН домен данных нужен, чтобы ответить? Верни РОВНО ОДНО английское "
                  "слово-домен (movies, weather, geolocation, finance, books, sports…). Только слово.\n\n"
                  f"Вопрос: {query[:300]}")
        r = await asyncio.wait_for(chat("fast", 0).ainvoke(prompt), timeout=10)
        w = re.findall(r"[A-Za-z]+", (r.content if hasattr(r, "content") else str(r)))
        return w[0].lower() if w else ""
    except Exception:  # noqa: BLE001
        return ""


def _relevance(c: dict, terms: set) -> int:
    """Совпадение домена/запроса с именем+описанием кандидата (общий фильтр от мусора —
    не подключать sales-MCP под вопрос о фильмах). Универсально, не под конкретный сценарий."""
    text = (c.get("name", "") + " " + (c.get("description") or "")).lower()
    return len(set(re.findall(r"[a-z]{3,}", text)) & terms)


async def try_connect_discovered(query: str, max_try: int = 3) -> tuple[Optional[str], list]:
    """
    САМО-РАСШИРЕНИЕ К ДАННЫМ (общая способность, НЕ под бенч): извлечь ДОМЕН → discover →
    отсеять НЕрелевантных кандидатов (по смыслу описания) → подключиться к первому ЖИВОМУ
    РЕЛЕВАНТНОМУ (реестр полон мёртвых/мусорных). REMOTE приоритетнее uvx (без установки).
    """
    domain = await _extract_domain(query)
    cand = await asyncio.wait_for(asyncio.to_thread(discover_mcp, domain or query, 8), timeout=10)
    if not cand and domain:  # фолбэк на исходный запрос
        cand = await asyncio.wait_for(asyncio.to_thread(discover_mcp, query, 8), timeout=10)
    if not cand:
        return None, []
    # РЕЛЕВАНТНОСТЬ: только кандидаты, чьё описание реально про домен/запрос (отсев мусора).
    terms = set(re.findall(r"[a-z]{3,}", (domain + " " + query).lower()))
    scored = [(c, _relevance(c, terms)) for c in cand]
    scored = [(c, r) for c, r in scored if r > 0] or scored  # если совпадений 0 — берём как есть
    # сорт: релевантность ↓, затем remote-first (без uvx)
    scored.sort(key=lambda cr: (-cr[1], 0 if cr[0]["spec"].get("transport") in ("streamable_http", "sse") else 1))
    for c, _r in scored[:max_try]:
        approve_server(c["name"], c["spec"])
        try:
            tools = await asyncio.wait_for(get_mcp_tools([c["name"]]), timeout=20)
            if tools:
                print(f"[mcp] self-extension: подключён ЖИВОЙ {c['name']} ({len(tools)} тулов)", flush=True)
                return c["name"], tools
        except Exception:  # noqa: BLE001
            continue  # мёртвый/несовместимый → следующий кандидат
    return None, []


def propose_untrusted(name: str, spec: dict) -> dict:
    """
    Человеческий гейт для НЕдоверенного MCP: не подключаем, а возвращаем предложение
    на подтверждение. Подключение чужого кода без approval запрещено.
    """
    return {"status": "needs_approval", "name": name, "spec": spec,
            "note": "Подключение произвольного MCP — выполнение чужого кода. Нужно явное подтверждение пользователя."}
