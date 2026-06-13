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


def main() -> None:
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

    webview.create_window("self-extension-agent", f"http://{HOST}:{port}/",
                          width=980, height=760, min_size=(560, 480))
    webview.start()  # блокирует до закрытия окна; daemon-поток сервера завершится с процессом


if __name__ == "__main__":
    main()
