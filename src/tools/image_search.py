"""
Поиск картинок для показа В ЧАТЕ (desktop-GUI рендерит markdown ![](url) как картинку).

Бэкенды (graceful, по убыванию надёжности):
  1. SearXNG категория images (если задан SEARXNG_URL) — даёт прямой img_src + источник;
  2. DuckDuckGo i.js (vqd-токен со страницы) — фолбэк без своего инстанса.

Возвращает готовый markdown-блок ![подпись](img_url) — агент вставляет его прямо в ответ,
и десктоп показывает картинки. URL'ы РЕАЛЬНЫЕ из выдачи (не выдуманные) → проходят anti-URL
фильтр как заземлённые; синтаксис картинок вдобавок исключён из фильтра явно.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

_UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _searxng_base() -> str:
    """SEARXNG_URL читаем В МОМЕНТ ВЫЗОВА (а не снапшотом при импорте) — устойчиво к порядку
    импорта/позднему load_dotenv. Тот же инстанс, что и веб-поиск."""
    return os.getenv("SEARXNG_URL", "").rstrip("/")


def _searxng_images(query: str, n: int) -> list[dict]:
    """SearXNG image-категория → [{title, img, source}]. Пусто, если инстанс не задан/лёг."""
    base = _searxng_base()
    if not base:
        return []
    params = {"q": query, "format": "json", "categories": "images"}
    url = f"{base}/search?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8", "ignore"))
    out = []
    for r in data.get("results", []):
        img = r.get("img_src") or r.get("thumbnail_src") or ""
        if img.startswith("//"):
            img = "https:" + img
        if img.startswith("http"):
            out.append({"title": (r.get("title") or "").strip()[:120],
                        "img": img, "source": r.get("url", "")})
        if len(out) >= n:
            break
    return out


def _ddg_images(query: str, n: int) -> list[dict]:
    """DuckDuckGo image-поиск: vqd-токен со страницы → i.js JSON. Фолбэк без SearXNG."""
    q = urllib.parse.quote(query)
    page = urllib.request.Request(f"https://duckduckgo.com/?q={q}&iax=images&ia=images",
                                  headers={"User-Agent": _UA["User-Agent"]})
    with urllib.request.urlopen(page, timeout=15) as resp:
        html = resp.read().decode("utf-8", "ignore")
    m = re.search(r"vqd=[\"']?([\d-]+)", html)
    if not m:
        return []
    api = (f"https://duckduckgo.com/i.js?l=ru-ru&o=json&q={q}"
           f"&vqd={m.group(1)}&f=,,,&p=1")
    req = urllib.request.Request(api, headers={**_UA, "Referer": "https://duckduckgo.com/"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8", "ignore"))
    out = []
    for r in data.get("results", []):
        img = r.get("image") or r.get("thumbnail") or ""
        if img.startswith("http"):
            out.append({"title": (r.get("title") or "").strip()[:120],
                        "img": img, "source": r.get("url", "")})
        if len(out) >= n:
            break
    return out


def search_images(query: str, count: int = 4) -> str:
    """
    Find images on the web and return them as MARKDOWN so they SHOW in the desktop chat.
    Use when the user asks to SEE/show pictures, photos, diagrams, examples, "как выглядит …".
    Returns ready ![caption](url) blocks — put them straight into your answer to display them.
    """
    n = max(1, min(int(count or 4), 8))
    items: list[dict] = []
    for backend in (_searxng_images, _ddg_images):
        try:
            items = backend(query, n)
            if items:
                break
        except Exception:  # noqa: BLE001 — провайдер лёг → следующий
            continue
    if not items:
        return ("[поиск картинок недоступен: ни SearXNG (SEARXNG_URL), ни DuckDuckGo не ответили. "
                "Опиши словами или предложи пользователю уточнить запрос.]")
    # Структурный блок-ГАЛЕРЕЯ: десктоп-GUI рендерит его как отдельное «окно» с картинками.
    # Каждая строка: img_url ||| подпись ||| источник. Агент кладёт блок КАК ЕСТЬ и пишет свой
    # комментарий СВЕРХУ (что это) и СНИЗУ (вывод). Блок исключён из anti-URL фильтра целиком.
    lines = ["```sea-gallery", f"# {query}"]
    for it in items:
        cap = (it["title"] or query).replace("|||", "/").replace("\n", " ").strip()
        lines.append(f"{it['img']} ||| {cap} ||| {it.get('source', '')}")
    lines.append("```")
    block = "\n".join(lines)
    return (f"Нашёл {len(items)} изобр. по «{query}». Вставь блок В ОТВЕТ как есть, добавив "
            f"свой комментарий ДО окна (что на картинках) и ПОСЛЕ (вывод/детали):\n\n{block}")


def make_image_search_tool() -> StructuredTool:
    class _Q(BaseModel):
        query: str = Field(description="What to find images of, e.g. 'красная панда', 'transformer architecture diagram'")
        count: int = Field(default=4, description="How many images (1-8, default 4)")

    return StructuredTool.from_function(
        func=search_images, name="search_images", args_schema=_Q,
        description=search_images.__doc__,
    )
