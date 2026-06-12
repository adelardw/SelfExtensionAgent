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


async def _route(op: str, *args):
    """Физический веб = РАСШИРЕНИЕ (твой браузер). Нет расширения → просим поставить.
    Песочное окно (playwright) — только если юзер ЯВНО выбрал его: cli.browser_backend='window'."""
    br = _bridge()
    br.ensure_server()  # идемпотентно: гарантируем, что мост поднят (REPL/бот/сервер)
    if br.connected():
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


@tool
async def browser_open(url: str) -> str:
    """Open a URL in YOUR browser (extension: your logins/tabs; else a controlled window).
    Returns a snapshot: numbered interactive elements to click/type into.

    Args:
        url: The SPECIFIC service page, e.g. music.yandex.ru/search?text=... or youtube.com/results?search_query=...
    """
    return await _route("open_url", url)


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
