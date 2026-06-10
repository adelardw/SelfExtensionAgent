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
from concurrent.futures import ThreadPoolExecutor
from langchain_core.tools import tool

_BROWSE_HARD_TIMEOUT = 35  # сек: жёсткий потолок одного браузерного чтения (анти-зависание)

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
                # cloak-поиск тоже в потоке с жёстким таймаутом (sync-Playwright может зависнуть)
                with ThreadPoolExecutor(max_workers=1) as ex:
                    results = ex.submit(_search_cloak, query, max_results, recency).result(timeout=_BROWSE_HARD_TIMEOUT)
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


def _relevant_chunks(text: str, find: str, budget: int = 3500) -> str:
    """
    Прицельная выжимка: режем текст на абзацы, ранжируем по пересечению с запросом
    (token-overlap, без LLM — дёшево) и собираем самые релевантные в пределах budget.
    Так агент получает ИМЕННО нужный факт, а не 50к символов шума.
    """
    qtokens = {w for w in re.findall(r"\w+", find.lower()) if len(w) > 2}
    if not qtokens:
        return text[:budget]
    chunks = [c.strip() for c in re.split(r"\n{2,}|\. (?=[A-ZА-Я])", text) if len(c.strip()) > 30]
    scored = []
    for i, c in enumerate(chunks):
        ctokens = set(re.findall(r"\w+", c.lower()))
        overlap = len(qtokens & ctokens)
        if overlap:
            # бонус за плотность совпадений и за наличие чисел (факты часто числовые)
            density = overlap / (1 + len(ctokens) ** 0.5)
            num_bonus = 0.5 if re.search(r"\d", c) else 0
            scored.append((overlap + density + num_bonus, i, c))
    scored.sort(reverse=True)
    out, used = [], 0
    for _, i, c in scored:
        if used + len(c) > budget:
            continue
        out.append((i, c))
        used += len(c) + 2
        if used >= budget:
            break
    out.sort()  # вернуть в исходном порядке (связность)
    return "\n\n".join(c for _, c in out) if out else text[:budget]


def _page_text_urllib(url: str) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", "ignore")
    title = _clean((re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I) or [None, ""])[1])
    body = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    return title, _clean(re.sub("<[^>]+>", " ", body))


def _page_text_cloak(url: str) -> tuple[str, str]:
    browser = launch(headless=True)
    try:
        page = browser.new_page()
        page.goto(url, timeout=25000, wait_until="domcontentloaded")
        title = _clean(page.title())
        el = page.query_selector("article") or page.query_selector("main") or page.query_selector("body")
        return title, (_clean(el.inner_text()) if el else "")
    finally:
        browser.close()


def _page_text(url: str) -> tuple[str, str]:
    """
    (title, ПОЛНЫЙ чистый текст). cloakbrowser в ОТДЕЛЬНОМ ПОТОКЕ с жёстким таймаутом:
    sync-Playwright может ЗАВИСНУТЬ и заблокировать event loop так, что внешний
    asyncio.wait_for его не отменит (eval ловил прогоны на 240с). future.result(timeout)
    даёт нам «бросить» зависший браузер и уйти на urllib, не дожидаясь его.
    """
    if _CLOAK:
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                title, text = ex.submit(_page_text_cloak, url).result(timeout=_BROWSE_HARD_TIMEOUT)
            if text:
                return title, text
        except Exception:  # noqa: BLE001 — таймаут/сбой браузера → простой ридер
            pass
    return _page_text_urllib(url)


@tool
def browse(url: str, find: str = "") -> str:
    """
    Open a page and read it for a FACT. Reads the WHOLE page (not a 4k slice) and, when
    `find` is given, returns the most RELEVANT passages to that query — use this for
    precise fact-finding (counts, dates, names, numbers) after search_web.

    Args:
        url: Page URL to read.
        find: What to look for on the page (e.g. 'number of studio albums 2000-2009').
              Strongly recommended — returns targeted passages instead of raw dump.

    Returns:
        Title + targeted relevant text (or a longer extract if `find` is empty).
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        title, text = _page_text(url)
        if not text:
            return f"Страница {url} пустая или не отрендерилась."
        body = _relevant_chunks(text, find, budget=3500) if find else text[:5000]
        head = f"# {title}\n{url}" + (f"\n[поиск: {find}]" if find else "")
        return f"{head}\n\n{body}"
    except Exception as e:  # noqa: BLE001
        return f"Не удалось прочитать {url}: {type(e).__name__}: {e}"


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
