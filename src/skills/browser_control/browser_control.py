"""
browser_control — действия в браузере. ДВА бэкенда, выбор автоматический:
  • если подключено браузерное РАСШИРЕНИЕ (агент живёт в ТВОЁМ браузере: твои логины,
    твои вкладки, без перелогина) — команды идут туда;
  • иначе фолбэк на управляемое окно (src.browser_session, playwright).
Поверхность одинаковая. Живое состояние держат core-модули (навык перезагружается
на каждом подключении тулов).
"""
import asyncio

from langchain_core.tools import tool


def _bridge():
    from src import browser_bridge
    return browser_bridge


def _session():
    from src import browser_session
    return browser_session


def _install_hint() -> str:
    """Физический веб требует расширения (агент в ТВОЁМ браузере). Нет его → просим
    поставить, НЕ открываем песочное окно (это не «твой мир», юзер это отверг)."""
    from src import browser_bridge
    return (
        "[НУЖНО РАСШИРЕНИЕ] Чтобы действовать в ТВОЁМ браузере (твои логины, твои вкладки), "
        "поставь расширение агента ОДИН раз: chrome://extensions → Developer mode → "
        "Load unpacked → папка extension/ ; в попапе расширения вставь токен "
        f"{browser_bridge.token()} . После этого повтори просьбу. "
        "Сообщи это пользователю через ask_user и НЕ пытайся сделать это действие иначе "
        "(подпроцессом, поиском, отдельным окном) — физический веб только через расширение."
    )


def _backend() -> str:
    try:
        from src.cli_config import get_cli
        return (get_cli("browser_backend") or "extension").lower()
    except Exception:  # noqa: BLE001
        return "extension"


# PUPPETEER-БЭКЕНД: open/see идут через pp_* (Puppeteer ЖДЁТ оседания тяжёлого SPA, потом наш
# снапшот на отрендеренной странице — видит динамику, которую снапшот-без-ожидания пропускал;
# живой провал: карточки товаров Я.Еды). Клики/ввод/медиа — по тому же снапшоту, без изменений.
_PP_OP = {"open_url": "pp_open", "see": "pp_see"}


async def _route(op: str, *args):
    """Физический веб = РАСШИРЕНИЕ (твой браузер). Нет расширения → просим поставить.
    Песочное окно (playwright) — только если юзер ЯВНО выбрал его: cli.browser_backend='window'."""
    br = _bridge()
    br.ensure_server()  # идемпотентно: гарантируем, что мост поднят (REPL/бот/сервер)
    if br.connected():
        # backend=puppeteer|hybrid: open/see через Puppeteer-ожидание (тяжёлые SPA). Иначе как было.
        if _backend() in ("puppeteer", "hybrid") and op in _PP_OP:
            kw = {"url": args[0]} if args else {}
            for attempt in range(2):  # SW мог дёрнуться/заснуть → один ретрай ДО фолбэка
                try:
                    res = await br._send(_PP_OP[op], **kw)
                    # «не подключено»/ошибка моста — это НЕ результат страницы: ретрай/фолбэк,
                    # НЕ отдаём слепой снапшот (он соврёт «товаров нет», хотя pptr их видит).
                    if isinstance(res, str) and not res.startswith(("[расширение", "pp ошибка")):
                        return res
                except Exception:  # noqa: BLE001
                    pass
                if attempt == 0:
                    await asyncio.sleep(1.5)  # дать SW/WS переподключиться
            # оба раза pptr не дал страницу → обычный снапшот как последний шанс
        return await getattr(br, op)(*args)
    backend = "extension"
    try:
        from src.cli_config import get_cli
        backend = (get_cli("browser_backend") or "extension").lower()
    except Exception:  # noqa: BLE001
        pass
    if backend == "window":  # осознанный выбор power-user'а — управляемое окно
        return await asyncio.to_thread(getattr(_session(), op), *args)
    # Браузер закрыт/расширение спит → ПОДНЯТЬ браузер и дождаться автоподключения
    # расширения (юзер просил: при закрытом браузере расширение подтягивается само).
    if await asyncio.to_thread(br.launch_browser):
        if await asyncio.to_thread(br.wait_connected, 8.0):
            return await getattr(br, op)(*args)
    return _install_hint()  # расширение реально не установлено


def _looks_404(snapshot: str) -> bool:
    """Открытая страница — ошибка/404 (часто из-за угаданного URL поиска)."""
    head = (snapshot or "")[:300].lower()
    return bool(__import__("re").search(r"\b404\b|страница не найдена|page not found|"
                                        r"не найден[ао]|ничего не нашлось", head))


@tool
async def browser_open(url: str) -> str:
    """Open the service's page in YOUR browser (extension: your logins/tabs).
    Returns a snapshot: numbered interactive elements to click/type into.

    PREFER opening the service's HOME (just its domain), then use browser_see + browser_type
    into its on-page search box. Do NOT hand-craft search URLs (`/search?q=`, `/index.php?...`)
    — formats differ per site and often 404. If a URL 404s, this tool auto-falls back to the
    domain home so you can search there.

    Args:
        url: The service page to open (its domain/home, or a known content page). Pick the
            service that suits the user/domain/region; don't default to one brand.
    """
    res = await _route("open_url", url)
    # ДЕТЕРМИНИРОВАННАЯ 404-RECOVERY: угаданный search-URL → 404. Не полагаемся на модель —
    # сами открываем ГЛАВНУЮ домена (там есть строка поиска), возвращаем её снапшот с подсказкой.
    if isinstance(res, str) and _looks_404(res):
        import urllib.parse as _up
        p = _up.urlsplit(url if "://" in url else "https://" + url)
        root = f"{p.scheme or 'https'}://{p.netloc}/"
        if p.netloc and root.rstrip("/") != (url.rstrip("/")):
            home = await _route("open_url", root)
            if isinstance(home, str) and not _looks_404(home):
                return (f"[URL {url} дал 404 — открыл главную {root}. Найди тут строку поиска "
                        f"(browser_see) и введи запрос через browser_type, не угадывай URL.]\n{home}")
    return res


@tool
async def browser_see() -> str:
    """Look at the current page: numbered list of interactive elements (links, buttons,
    inputs, players) + whether sound is playing. Use the numbers with click/type."""
    return await _route("see")


@tool
async def browser_click(item: int) -> str:
    """Click element by its number from the snapshot (track row, play button, video).

    Args:
        item: Element number, e.g. 7.
    """
    return await _route("click", int(item))


@tool
async def browser_type(item: int, text: str, submit: bool = True) -> str:
    """Type text into an input by its number and press Enter. Refuses password/card/ID fields.

    Args:
        item: Input element number from the snapshot.
        text: What to type (e.g. track/video name).
        submit: Press Enter after typing (default True).
    """
    return await _route("type_into", int(item), text, bool(submit))


@tool
async def browser_tap(item: int) -> str:
    """TRUSTED-click an element by its snapshot number (real gesture via CDP). Use when a normal
    browser_click did NOT react — React/SPA buttons (e.g. «+»/«В корзину» on Я.Еда/Лавка) often
    ignore a plain DOM click and need a real tap.

    Args:
        item: Element number from the snapshot (the «+»/add button).
    """
    return await _route("tap", int(item))


@tool
async def browser_click_text(text: str) -> str:
    """Click an element by its VISIBLE TEXT — use when the numbered snapshot does NOT list the
    item you need (dynamic dropdowns/overlays of SPA sites: autocomplete results, custom menus).
    Finds the most specific visible element containing that text and clicks its clickable parent.

    Args:
        text: The exact visible text of the item to click (e.g. a search-result title).
    """
    return await _route("click_text", text)


@tool
async def browser_media(action: str = "toggle") -> str:
    """Control ANY playing media on the page: pause | play | toggle | mute | unmute.
    Use when the user asks пауза/продолжи/выключи звук.

    Args:
        action: pause | play | toggle | mute | unmute (default toggle).
    """
    return await _route("media", action)


@tool
async def browser_read() -> str:
    """Read the visible TEXT of the current page — for «расскажи вкратце что играет» /
    «порекомендуй» / extract info from the page."""
    return await _route("read")


@tool
async def browser_press(key: str) -> str:
    """Press a keyboard key on the page (Enter, Escape, Space — pause/play video…).

    Args:
        key: Key name.
    """
    return await _route("press", key)


@tool
async def browser_scroll(direction: str = "down") -> str:
    """Scroll the page to reveal more elements. direction: down | up.

    Args:
        direction: 'down' (default) or 'up'.
    """
    return await _route("scroll", direction)
