"""
Десктоп-приложение: нативное окно с чат-GUI поверх FastAPI-мозга.

    uv sync --group gui          # поставить pywebview
    python desktop.py            # открыть окно

Архитектура «тонкий клиент + мозг»: uvicorn (граф агента) поднимается в фоновом
потоке на localhost, pywebview открывает нативное окно ОС (системный webview —
без Electron/Node), указывающее на встроенный веб-GUI (src/webui/index.html).
Кроссплатформенно: macOS (WebKit), Windows (WebView2/EdgeHTML), Linux (WebKitGTK).
"""
from __future__ import annotations

import multiprocessing
multiprocessing.freeze_support()  # PyInstaller: spawn-дочерний иначе РЕ-ИСПОЛНЯЕТ .app (перезапуск)

import socket
import threading
import time
import urllib.request

HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _serve(port: int) -> None:
    import uvicorn

    from src.server import app
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


def _wait_up(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://{HOST}:{port}/"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.4)
    return False


class _Api:
    """JS-мост pywebview (window.pywebview.api): нативные вещи, недоступные WKWebView из веба.
    pick_files — нативный файл-диалог ОС (HTML <input type=file> в WKWebView по программному клику
    не открывается). Возвращает ЛОКАЛЬНЫЕ пути; их сервер (та же машина) читает сам."""

    def pick_files(self):
        import webview
        try:
            res = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True)
            return list(res) if res else []
        except Exception:  # noqa: BLE001
            return []


def main() -> None:
    # УПАКОВАННОЕ приложение: при запуске из Launchpad/Dock cwd='/' (read-only) → относительные
    # data/, config.local.yml падают. Переходим в writable per-user папку ДО старта сервера/импортов.
    from src.config_paths import bootstrap_frozen
    bootstrap_frozen()
    try:
        import webview
    except ImportError:
        raise SystemExit("pywebview не установлен. Поставь: uv sync --group gui "
                         "(или: открой http://127.0.0.1:8000/ после `uvicorn src.server:app`)")

    port = _free_port()
    threading.Thread(target=_serve, args=(port,), daemon=True).start()
    print(f"🧠 мозг поднимается на http://{HOST}:{port} …")
    if not _wait_up(port):
        raise SystemExit("сервер не поднялся за отведённое время")

    webview.create_window("SEA — self-extension-agent", f"http://{HOST}:{port}/",
                          width=980, height=760, min_size=(560, 480), js_api=_Api())
    webview.start()  # блокирует до закрытия окна; daemon-поток сервера завершится с процессом


if __name__ == "__main__":
    main()
