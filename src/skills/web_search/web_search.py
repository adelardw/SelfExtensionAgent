"""
web_search — нормальный веб-поиск и чтение страниц через stealth-браузер
(cloakbrowser, drop-in замена Playwright, проходит антибот-детект).

Защищённый core-навык: search_web (выдача DuckDuckGo) + read_url (полный текст
страницы для agentic deep-read). Используется sync Playwright API — корректно
работает, так как агент вызывает синхронные tools в отдельном потоке executor'а.

Graceful: если cloakbrowser/Chromium недоступны — падает на urllib-фолбэк,
а не роняет навык.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
from langchain_core.tools import tool

try:
    from cloakbrowser import launch

    _CLOAK = True
except Exception:  # noqa: BLE001
    _CLOAK = False

# SearXNG — приватный self-host метапоиск ($0 API, свежесть через time_range,
# 200+ источников). Указывается через env SEARXNG_URL (напр. http://localhost:8080).
_SEARXNG = os.getenv("SEARXNG_URL", "").rstrip("/")
_RECENCY_RANGE = {"d": "day", "w": "week", "m": "month", "y": "year"}
_SEARXNG_COOLDOWN_S = 600          # после отказа не дёргаем SearXNG N секунд (анти-спам в лог)

# Cooldown живёт в env, а не в глобале: модуль навыка ПЕРЕЗАГРУЖАЕТСЯ на каждом
# подключении тулов (exec_module), и глобал сбрасывался бы каждый шаг.
def _searxng_down_until() -> float:
    try:
        return float(os.environ.get("SEARXNG_DOWN_UNTIL", "0"))
    except ValueError:
        return 0.0


def _set_searxng_down(until: float) -> None:
    os.environ["SEARXNG_DOWN_UNTIL"] = str(until)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _ddg_redirect(href: str) -> str:
    """DuckDuckGo html отдаёт ссылки через /l/?uddg=<urlencoded> — раскручиваем."""
    if href and "uddg=" in href:
        try:
            return urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
        except Exception:  # noqa: BLE001
            return href
    if href.startswith("//"):
        return "https:" + href
    return href


def _freshness(recency: str) -> str:
    """DuckDuckGo df-параметр: d=день, w=неделя, m=месяц, y=год."""
    return f"&df={recency}" if recency in ("d", "w", "m", "y") else ""


def _search_searxng(query: str, max_results: int, recency: str = "") -> list[dict]:
    """Поиск через SearXNG JSON API (приоритетный бэкенд, если задан SEARXNG_URL)."""
    params = {"q": query, "format": "json"}
    if recency in _RECENCY_RANGE:
        params["time_range"] = _RECENCY_RANGE[recency]
    url = f"{_SEARXNG}/search?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8", "ignore"))
    out = []
    for r in data.get("results", [])[:max_results]:
        out.append({
            "title": _clean(r.get("title", "")),
            "url": r.get("url", ""),
            "snippet": _clean(r.get("content", "")),
        })
    return out


def _search_cloak(query: str, max_results: int, recency: str = "") -> list[dict]:
    browser = launch(headless=True)
    try:
        page = browser.new_page()
        url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}{_freshness(recency)}"
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        out = []
        for el in page.query_selector_all("div.result")[: max_results + 4]:
            a = el.query_selector("a.result__a")
            if not a:
                continue
            snippet_el = el.query_selector("a.result__snippet") or el.query_selector(".result__snippet")
            out.append({
                "title": _clean(a.inner_text()),
                "url": _ddg_redirect(a.get_attribute("href") or ""),
                "snippet": _clean(snippet_el.inner_text()) if snippet_el else "",
            })
            if len(out) >= max_results:
                break
        return out
    finally:
        browser.close()


def _search_fallback(query: str, max_results: int, recency: str = "") -> list[dict]:
    """urllib-фолбэк через html.duckduckgo.com (без браузера)."""
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query) + _freshness(recency),
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", "ignore")
    out = []
    for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        out.append({
            "title": _clean(re.sub("<[^>]+>", "", m.group(2))),
            "url": _ddg_redirect(m.group(1)),
            "snippet": "",
        })
        if len(out) >= max_results:
            break
    return out


@tool
def search_web(query: str, max_results: int = 8, recency: str = "") -> str:
    """
    Search the web and return ranked results (title, url, snippet).
    Uses a stealth browser to avoid bot-blocking; falls back to plain HTTP.

    Args:
        query: Search query.
        max_results: How many results to return (default 8).
        recency: Freshness filter — '' (any), 'd' (past day), 'w' (week), 'm' (month), 'y' (year).
            Use 'd'/'w' for news and anything time-sensitive to get FRESH results.

    Returns:
        Formatted list of results, or an error string.
    """
    # Приоритет: SearXNG (приватный/свежий) → cloakbrowser (stealth) → urllib.
    engine = "http-fallback"
    results = []
    try:
        if _SEARXNG and time.time() >= _searxng_down_until():
            results = _search_searxng(query, max_results, recency)
            engine = "searxng"
    except Exception as e:  # noqa: BLE001
        # SearXNG недоступен (не поднят docker) → молчаливый cooldown, лог один раз,
        # а не спам "Connection refused" на каждый поисковый вызов.
        _set_searxng_down(time.time() + _SEARXNG_COOLDOWN_S)
        print(f"[web_search] SearXNG недоступен ({e}) — выключаю на {_SEARXNG_COOLDOWN_S // 60} мин, иду в fallback")

    try:
        if not results:
            # Сначала лёгкий urllib (тот же DDG, без запуска Chromium — браузер
            # «мигает» в доке на каждый вызов); cloak — только последняя надежда.
            try:
                results = _search_fallback(query, max_results, recency)
                engine = "http-fallback"
            except Exception:  # noqa: BLE001
                results = []
            if not results and _CLOAK:
                results = _search_cloak(query, max_results, recency)
                engine = "cloakbrowser"
        if not results:
            return f"Ничего не найдено по запросу: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}".rstrip())
        return f"Результаты поиска [{engine}] по «{query}»:\n\n" + "\n\n".join(lines)
    except Exception as e:  # noqa: BLE001
        try:
            results = _search_fallback(query, max_results)
            lines = [f"{i}. {r['title']}\n   {r['url']}" for i, r in enumerate(results, 1)]
            return "Результаты (fallback):\n" + "\n".join(lines) if lines else f"Ошибка поиска: {e}"
        except Exception as e2:  # noqa: BLE001
            return f"Ошибка поиска: {e}; fallback: {e2}"


@tool
def read_url(url: str, max_chars: int = 4000) -> str:
    """
    Open a URL in a stealth browser and return its readable text content.
    Use for deep reading of a specific page (agentic RAG step after search_web).

    Args:
        url: Page URL to read.
        max_chars: Truncate extracted text to this many characters (default 4000).

    Returns:
        Extracted page text, or an error string.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        if _CLOAK:
            browser = launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                title = _clean(page.title())
                body = page.query_selector("body")
                text = _clean(body.inner_text()) if body else ""
                return f"# {title}\n{url}\n\n{text[:max_chars]}"
            finally:
                browser.close()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", "ignore")
        text = _clean(re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I))
        text = _clean(re.sub("<[^>]+>", " ", text))
        return f"{url}\n\n{text[:max_chars]}"
    except Exception as e:  # noqa: BLE001
        return f"Не удалось прочитать {url}: {e}"
