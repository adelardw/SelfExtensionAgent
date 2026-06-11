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


def _run_bounded(fn, *args, timeout):
    """
    Sync fn в отдельном потоке с ЖЁСТКИМ таймаутом, БЕЗ ожидания зависшего потока.
    КРИТИЧНО: раньше использовался `with ThreadPoolExecutor() as ex:` — его __exit__ зовёт
    shutdown(wait=True) и БЛОКИРУЕТ прогон НАВСЕГДА, если поток залип на Chromium (sync
    Playwright). result(timeout) бросал таймаут, а shutdown всё равно ждал → дедлок всего
    агента (event loop idle, asyncio-executor исчерпан). shutdown(wait=False) — бросаем поток.
    """
    from concurrent.futures import ThreadPoolExecutor as _TPE
    ex = _TPE(max_workers=1)
    fut = ex.submit(fn, *args)
    try:
        return fut.result(timeout=timeout)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

try:
    from cloakbrowser import launch

    _CLOAK = True
except Exception:  # noqa: BLE001
    _CLOAK = False

# cloakbrowser спавнит Chromium-подпроцессы, которые накапливаются/виснут (leaked semaphores,
# asyncio _do_waitpid) и давали 240с-таймауты в eval (гарантированные 0 на L2/L3, $0.0011 =
# блокировка в sync, не работа). В eval/без-браузер режиме отключаем cloak полностью —
# SearXNG/urllib дают факты без Chromium. AGENT_NO_BROWSER=1 — то же для прода при желании.
if os.getenv("AGENT_EVAL_MODE") == "1" or os.getenv("AGENT_NO_BROWSER") == "1":
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
    """
    urllib-фолбэк через DuckDuckGo html (без браузера). ВАЖНО: GET (?q=) теперь БЛОКИРУЕТСЯ
    антиботом DDG (отдаёт пустую страницу) — используем POST (form data), он работает и
    возвращает результаты. Это надёжный поиск БЕЗ cloak/Chromium, не зависит от флакающего
    SearXNG (раньше при падении SearXNG поиск отдавал 0 → research не стартовал).
    """
    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/", data=data,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", "ignore")
    out = []
    # Основной паттерн результата DDG.
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        out.append({"title": _clean(re.sub("<[^>]+>", "", m.group(2))),
                    "url": _ddg_redirect(m.group(1)), "snippet": ""})
        if len(out) >= max_results:
            break
    if not out:  # фолбэк: прямые внешние ссылки (разметка DDG могла измениться)
        seen = set()
        for u in re.findall(r'href="(https?://[^"]+)"', html):
            if "duckduckgo.com" in u or u in seen:
                continue
            seen.add(u)
            out.append({"title": "", "url": u, "snippet": ""})
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
                # cloak-поиск в потоке с жёстким таймаутом БЕЗ ожидания зависшего потока (дедлок)
                results = _run_bounded(_search_cloak, query, max_results, recency, timeout=_BROWSE_HARD_TIMEOUT)
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


# ── Контекстный инжиниринг ПОИСКА: полную страницу НИКОГДА не кормим агенту ──────
# Пайплайн (grounded в исследовании, см. CLAUDE.md): чистка (trafilatura) → чанкинг →
# BM25S (лексический отсев, до 500× быстрее rank_bm25) → vector-rerank (OpenRouter
# embeddings, только топ-кандидатов = дёшево) → сборка в пределах budget. Каждый слой
# с graceful-фолбэком: нет bm25s → token-overlap; нет ключа эмбеддингов → только BM25S.
try:
    import bm25s
    _BM25 = True
except Exception:  # noqa: BLE001
    _BM25 = False

_CHUNK_CANDIDATES = 20   # сколько топ-чанков отдаём из BM25S на vector-rerank


def _chunk(text: str, target: int = 500) -> list[str]:
    """Связные куски ~target символов по границам абзацев/предложений."""
    chunks: list[str] = []
    for p in (p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 30):
        if len(p) <= target * 2:
            chunks.append(p)
            continue
        buf = ""
        for s in re.split(r"(?<=[.!?])\s+", p):
            if len(buf) + len(s) > target and buf:
                chunks.append(buf.strip())
                buf = s
            else:
                buf = f"{buf} {s}" if buf else s
        if buf.strip():
            chunks.append(buf.strip())
    return chunks or ([text[:target]] if text else [])


def _token_overlap_top(chunks: list[str], find: str, top: int) -> list[int]:
    """Фолбэк-ранкер (без BM25S): пересечение токенов + бонус за числа."""
    qtokens = {w for w in re.findall(r"\w+", find.lower()) if len(w) > 2}
    scored = []
    for i, c in enumerate(chunks):
        ctokens = set(re.findall(r"\w+", c.lower()))
        overlap = len(qtokens & ctokens)
        if overlap:
            density = overlap / (1 + len(ctokens) ** 0.5)
            scored.append((overlap + density + (0.5 if re.search(r"\d", c) else 0), i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:top]] or list(range(min(top, len(chunks))))


def _bm25_top(chunks: list[str], find: str, top: int) -> list[int]:
    """
    Лексический отсев BM25S → индексы РЕЛЕВАНТНЫХ кандидатов (score>0), в порядке ранга.
    Языко-агностичная токенизация (stopwords/stemmer off) — корректно для рус/смешанного.
    Нулевые кандидаты НЕ возвращаем, иначе бы они забивали budget шумом. Фолбэк — token-overlap.
    """
    if not _BM25:
        return _token_overlap_top(chunks, find, top)
    try:
        tok = dict(stopwords=None, stemmer=None, show_progress=False)
        retr = bm25s.BM25()
        retr.index(bm25s.tokenize(chunks, **tok), show_progress=False)
        results, scores = retr.retrieve(bm25s.tokenize(find, **tok),
                                        k=min(top, len(chunks)), show_progress=False)
        idx = [int(i) for i, sc in zip(results[0], scores[0]) if sc > 0]
        return idx or _token_overlap_top(chunks, find, top)
    except Exception:  # noqa: BLE001
        return _token_overlap_top(chunks, find, top)


def _embed_batch(texts: list[str]) -> list | None:
    """Батч-эмбеддинги через OpenRouter (openai/text-embedding-3-small) — один вызов."""
    key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        if os.getenv("OPEN_ROUTER_API_KEY"):
            client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=10, max_retries=0)
            model = "openai/text-embedding-3-small"
        else:
            client, model = OpenAI(api_key=key, timeout=10, max_retries=0), "text-embedding-3-small"
        resp = client.embeddings.create(model=model, input=[t[:4000] for t in texts])
        return [d.embedding for d in resp.data]
    except Exception:  # noqa: BLE001
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _relevant_chunks(text: str, find: str, budget: int = 3500) -> str:
    """
    Прицельная выжимка БЕЗ скармливания всей страницы: чанкинг → BM25S → vector-rerank →
    сборка в budget. В контекст агента попадают граммы релевантного, не килобайты шума.
    """
    if not find:
        return text[:budget]
    chunks = _chunk(text)
    if len(chunks) <= 1:
        return text[:budget]
    # 1) лексический отсев → кандидаты
    cand_idx = _bm25_top(chunks, find, _CHUNK_CANDIDATES)
    cand = [chunks[i] for i in cand_idx]
    # 2) семантический ре-ранк кандидатов (только их → дёшево); фолбэк — порядок BM25S
    order = list(range(len(cand)))
    vecs = _embed_batch([find] + cand)
    if vecs and len(vecs) == len(cand) + 1:
        sims = sorted(((_cosine(vecs[0], vecs[i + 1]), i) for i in range(len(cand))), reverse=True)
        order = [i for _, i in sims]
    # 3) собираем в пределах budget, возвращаем в порядке исходного текста (связность)
    picked, used = [], 0
    for oi in order:
        c = cand[oi]
        if used + len(c) > budget and picked:
            continue
        picked.append((cand_idx[oi], c))
        used += len(c) + 2
        if used >= budget:
            break
    picked.sort()
    return "\n\n".join(c for _, c in picked) if picked else text[:budget]


def _page_text_urllib(url: str) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", "ignore")
    title = _clean((re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I) or [None, ""])[1])
    # ЧИСТКА: trafilatura извлекает основной текст (убирает меню/футер/скрипты/реклама),
    # отдаёт читаемый контент с таблицами — намного чище наивного strip-тегов.
    try:
        import trafilatura
        extracted = trafilatura.extract(html, include_comments=False, include_tables=True,
                                         favor_recall=True)
        if extracted and len(extracted) > 100:
            return title, _clean(extracted)
    except Exception:  # noqa: BLE001
        pass
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


_FAST_ENOUGH = 600  # символов чистого текста с urllib-пути достаточно → не будим браузер


def _page_text(url: str) -> tuple[str, str]:
    """
    (title, чистый текст). БЫСТРЫЙ ПУТЬ ПЕРВЫМ: urllib + trafilatura (~0.8с, чистый текст
    с таблицами). cloakbrowser (Chromium, 2–35с, sync-Playwright МОЖЕТ ЗАВИСНУТЬ и
    заблокировать event loop) — ТОЛЬКО фолбэк, когда быстрый путь дал мало (бот-стена,
    JS-рендер). Раньше cloak шёл первым → лишние секунды и риск 35с-зависания на КАЖДОЙ
    странице. Браузер запускаем в отдельном потоке с жёстким таймаутом (можно «бросить»).
    """
    try:
        title, text = _page_text_urllib(url)
    except Exception:  # noqa: BLE001
        title, text = "", ""
    if len(text) >= _FAST_ENOUGH:
        return title, text
    # мало текста → вероятно бот-стена/JS → пробуем браузер (ограниченно по времени)
    if _CLOAK:
        try:
            c_title, c_text = _run_bounded(_page_text_cloak, url, timeout=_BROWSE_HARD_TIMEOUT)
            if len(c_text) > len(text):
                return c_title or title, c_text
        except Exception:  # noqa: BLE001 — таймаут/сбой браузера → что есть с urllib
            pass
    return title, text


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
