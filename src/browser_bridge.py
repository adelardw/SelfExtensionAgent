"""
Мост агент ↔ браузерное расширение (контур «агент живёт в ТВОЁМ браузере»).

Почему расширение, а не управляемый Chrome: оно работает в РЕАЛЬНОМ браузере пользователя,
в его профиле и сессии — твои логины уже там, перелогиниваться не надо, перезапускать/
закрывать браузер не надо, читать пароли/cookies не надо. Расширение действует в твоих же
вкладках (как любой нормальный браузерный ассистент).

Архитектура: агент поднимает локальный WebSocket-сервер (только 127.0.0.1). Фоновый
service worker расширения подключается к нему и ждёт команды. Инструменты browser_*
сериализуют команду → мост → расширение исполняет в активной/новой вкладке через
chrome.scripting → возвращает снапшот элементов. Тот же поверхностный API, что и у
playwright-бэкенда (open/see/click/type/media/read), но исполнитель — твой браузер.

Безопасность: сервер слушает ТОЛЬКО loopback; токен-рукопожатие (data/browser_bridge.token)
отсекает чужие подключения; никаких паролей/cookies мост не трогает — только DOM-команды.
"""
from __future__ import annotations

import asyncio
import json
import platform
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional

HOST = "127.0.0.1"
PORT = 8777
TOKEN_FILE = Path("data/browser_bridge.token")
LOG_FILE = Path("data/browser_bridge.log")  # трафик мост↔расширение (для диагностики плеера)
CMD_TIMEOUT = 25  # сек на исполнение команды в расширении (хватает с запасом)


_prev_app: Optional[str] = None


def _capture_front() -> None:
    """Запомнить активное приложение пользователя ДО браузерной команды (macOS)."""
    global _prev_app
    if platform.system() != "Darwin":
        return
    try:
        _prev_app = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to name of first process whose frontmost is true'],
            capture_output=True, text=True, timeout=3).stdout.strip() or None
    except Exception:  # noqa: BLE001
        _prev_app = None


def _restore_front() -> None:
    """Вернуть фокус приложению пользователя — вкладка агента работает в ФОНЕ, не
    перетягивает на себя (юзер продолжает свои дела). macOS."""
    if platform.system() != "Darwin" or not _prev_app or _prev_app == "Google Chrome":
        return
    try:
        subprocess.run(["osascript", "-e", f'tell application "{_prev_app}" to activate'],
                       capture_output=True, timeout=3)
    except Exception:  # noqa: BLE001
        pass


def _log(line: str) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
    except Exception:  # noqa: BLE001
        pass

_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_client = None  # текущее WS-соединение расширения (одно)
_pending: dict[str, asyncio.Future] = {}
# Обработчик чата ИЗ расширения (extension→agent): фронтенд (REPL/сервер) регистрирует
# функцию, которая прогоняет запрос через граф и возвращает ответ.
_chat_handler: Optional[Callable[[str], Awaitable[str]]] = None


def set_chat_handler(fn: Optional[Callable[[str], Awaitable[str]]]) -> None:
    global _chat_handler
    _chat_handler = fn


def token() -> str:
    """Стабильный per-машина токен рукопожатия (генерится один раз)."""
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    t = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(t)
    return t


async def _handler(ws):
    global _client
    # Рукопожатие: первое сообщение — токен.
    try:
        hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
    except Exception:  # noqa: BLE001
        return
    if hello.get("token") != token():
        _log("handshake: ОТКЛОНЁН (неверный токен)")
        await ws.close(code=4001, reason="bad token")
        return
    _client = ws
    _log("handshake: расширение подключено ✓")
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            if msg.get("type") == "ping":  # keepalive от расширения (держит WS/worker живым)
                continue
            # Чат ИЗ расширения (extension→agent): прогнать через граф, вернуть ответ.
            if msg.get("type") == "chat":
                asyncio.create_task(_serve_chat(ws, msg))
                continue
            # Иначе это ОТВЕТ на команду агента (agent→extension).
            fut = _pending.pop(msg.get("id", ""), None)
            if fut and not fut.done():
                fut.set_result(msg.get("result", ""))
    finally:
        if _client is ws:
            _client = None


async def _serve_chat(ws, msg) -> None:
    cmd_id = msg.get("id", "")
    text = (msg.get("text") or "").strip()
    if not _chat_handler or not text:
        await ws.send(json.dumps({"id": cmd_id, "result": "(чат недоступен)"}))
        return
    try:
        answer = await _chat_handler(text)
    except Exception as e:  # noqa: BLE001
        answer = f"(ошибка агента: {type(e).__name__})"
    await ws.send(json.dumps({"id": cmd_id, "result": answer}))


_serving = False  # реально ли наш сервер слушает порт


def _run_server():
    global _loop, _serving
    import websockets

    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def _serve():
        global _serving
        # reuse_address: пережить TIME_WAIT после рестарта агента (иначе bind падал).
        async with websockets.serve(_handler, HOST, PORT, reuse_address=True):
            _serving = True
            await asyncio.Future()  # вечно

    try:
        _loop.run_until_complete(_serve())
    except OSError as e:  # порт занят ДРУГИМ живым агентом — это ок, расширение к нему и цепляется
        _serving = False
        print(f"[bridge] порт {PORT} уже занят — вероятно, мост уже поднят другим процессом агента.")
    except Exception as e:  # noqa: BLE001
        _serving = False
        print(f"[bridge] сервер остановлен: {type(e).__name__}: {e}")


def _free_port() -> None:
    """Освободить порт 8777 от ЗАЛИПШИХ процессов (прежний main.py/тестовые прогоны).
    Порт выделен только этому приложению → всё на нём = устаревший экземпляр. Без этого
    расширение цеплялось к мёртвому процессу, а живой агент видел «не подключено»."""
    if platform.system() == "Windows":
        return
    try:
        out = subprocess.run(["lsof", "-ti", f"tcp:{PORT}"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        mypid = str(__import__("os").getpid())
        for pid in out.splitlines():
            if pid and pid != mypid:
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
    except Exception:  # noqa: BLE001
        pass


def ensure_server() -> None:
    """Поднять WS-сервер в фоновом потоке (идемпотентно)."""
    global _thread
    if _thread and _thread.is_alive() and _serving:
        return
    _free_port()       # отобрать порт у залипших экземпляров → мост принадлежит ЭТОМУ агенту
    token()            # сгенерировать файл токена для расширения
    _thread = threading.Thread(target=_run_server, daemon=True, name="browser-bridge")
    _thread.start()


def connected() -> bool:
    return _client is not None


def launch_browser() -> bool:
    """Поднять системный браузер пользователя (закрыт → откроется, расширение
    автозагрузится и подключится к мосту). True — команда запуска отправлена."""
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            # -g: открыть В ФОНЕ, не выносить Chrome на передний план (не красть фокус —
            # юзер продолжает свою работу, музыка играет фоном).
            subprocess.Popen(["open", "-g", "-a", "Google Chrome"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sysname == "Windows":
            subprocess.Popen(["cmd", "/c", "start", "chrome"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            opener = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("xdg-open")
            if not opener:
                return False
            subprocess.Popen([opener], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:  # noqa: BLE001
        return False


def wait_connected(timeout: float = 12.0) -> bool:
    """Подождать, пока расширение подключится к мосту (после запуска браузера)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if connected():
            return True
        time.sleep(0.4)
    return connected()


async def _send(cmd: str, **args) -> str:
    """Отправить команду расширению и дождаться результата (в его event loop).
    Параметр назван cmd (НЕ action): иначе media(action=…) давал коллизию имён."""
    if _loop is None or _client is None:
        return "[расширение не подключено]"
    cmd_id = uuid.uuid4().hex
    fut: asyncio.Future = _loop.create_future()
    _pending[cmd_id] = fut

    async def _do():
        await _client.send(json.dumps({"id": cmd_id, "action": cmd, "args": args}))
        return await asyncio.wait_for(fut, timeout=CMD_TIMEOUT)

    _log(f"→ {cmd} {args}")
    # Фон: запоминаем активное приложение юзера до команды, возвращаем фокус после —
    # вкладка агента не перетягивает на себя (живой баг: «открыл и переключил меня»).
    _capture_front()
    try:
        res = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_do(), _loop))
        _log(f"← {str(res)[:600]}")
        return res
    except Exception as e:  # noqa: BLE001
        _pending.pop(cmd_id, None)
        _log(f"← [не ответило: {type(e).__name__}]")
        return f"[расширение не ответило: {type(e).__name__}]"
    finally:
        await asyncio.to_thread(_restore_front)


# ── Поверхность, идентичная browser_session (исполнитель — расширение) ───────
async def open_url(url: str) -> str:
    return await _send("open", url=url)


async def see() -> str:
    return await _send("see")


async def click(item: int) -> str:
    return await _send("click", item=int(item))


async def type_into(item: int, text: str, submit: bool = True) -> str:
    return await _send("type", item=int(item), text=text, submit=bool(submit))


async def press(key: str) -> str:
    return await _send("press", key=key)


async def scroll(direction: str = "down") -> str:
    return await _send("scroll", direction=direction)


async def media(action: str = "toggle") -> str:
    return await _send("media", action=action)


async def read() -> str:
    return await _send("read")
