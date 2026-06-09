"""
device_control — действия с устройством по запросу из чата (on-demand, не мониторинг).

Принцип (как просил пользователь): агент НЕ анализирует в фоне, что открыто.
Только при обращении в чат он может открыть сайт/приложение, сделать скриншот
(чтобы «посмотреть» на экран), показать уведомление или озвучить ответ.

Реализация под macOS через штатные утилиты (open / screencapture / osascript / say) —
без сторонних зависимостей. Чувствительные действия (письма, звонки) — отдельный
слой с подтверждением; здесь только безопасные базовые операции.
"""
import os
import subprocess
import tempfile
import time
from langchain_core.tools import tool


def _run(cmd: list[str], timeout: int = 20) -> tuple[bool, str]:
    # Безопасный режим: AGENT_DRY_RUN=1 → не выполнять реальные действия с устройством.
    if os.getenv("AGENT_DRY_RUN"):
        return True, f"[dry-run] {' '.join(str(c) for c in cmd)[:200]}"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


@tool
def open_url(url: str) -> str:
    """Open a URL or web page in the default browser (on-demand).

    Args:
        url: Address to open (http/https).
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    ok, out = _run(["open", url])
    return f"Открыл {url}" if ok else f"Не удалось открыть: {out}"


@tool
def open_app(app_name: str) -> str:
    """Open a desktop application by name (e.g. 'Mail', 'Safari', 'Calendar').

    Args:
        app_name: Application name as shown in macOS.
    """
    ok, out = _run(["open", "-a", app_name])
    return f"Открыл приложение {app_name}" if ok else f"Не удалось открыть {app_name}: {out}"


@tool
def capture_screen() -> str:
    """Take a screenshot so the agent can 'look at' what is on screen right now.
    Returns the saved image path (for a follow-up multimodal/vision step).
    """
    path = f"{tempfile.gettempdir()}/agent_screen_{int(time.time())}.png"
    ok, out = _run(["screencapture", "-x", path])
    return f"Скриншот сохранён: {path}" if ok else f"Не удалось снять экран: {out}"


@tool
def notify(title: str, message: str) -> str:
    """Show a desktop notification about an activity.

    Args:
        title: Notification title.
        message: Notification body.
    """
    script = f'display notification "{message}" with title "{title}"'
    ok, out = _run(["osascript", "-e", script])
    return "Уведомление показано" if ok else f"Не удалось уведомить: {out}"


@tool
def speak(text: str) -> str:
    """Speak text aloud via the system voice (free local TTS fallback).
    For high-quality cloud TTS use OpenRouter's TTS endpoint instead.

    Args:
        text: Text to read aloud.
    """
    ok, out = _run(["say", text[:500]])
    return "Озвучил" if ok else f"Не удалось озвучить: {out}"


# ── работа с УЖЕ ОТКРЫТЫМИ приложениями (System Events; нужен Accessibility-доступ) ──

_KEYCODES = {"enter": 36, "return": 36, "tab": 48, "escape": 53, "esc": 53,
             "pagedown": 121, "pageup": 116, "down": 125, "up": 126, "space": 49, "home": 115, "end": 119}


def _osa(script: str) -> tuple[bool, str]:
    return _run(["osascript", "-e", script])


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


@tool
def scroll(direction: str = "down", amount: int = 5) -> str:
    """Scroll the currently focused window (e.g. the open browser) up or down.

    Args:
        direction: 'down' or 'up'.
        amount: Number of page scrolls.
    """
    code = _KEYCODES["pagedown"] if direction == "down" else _KEYCODES["pageup"]
    presses = "\n".join([f'key code {code}'] * max(1, min(amount, 30)))
    ok, out = _osa(f'tell application "System Events"\n{presses}\nend tell')
    return f"Проскроллил {direction} x{amount}" if ok else f"Не удалось: {out} (нужен Accessibility-доступ)"


@tool
def type_text(text: str) -> str:
    """Type text into the currently focused app/field (the active window).

    Args:
        text: Text to type.
    """
    ok, out = _osa(f'tell application "System Events" to keystroke "{_esc(text[:1000])}"')
    return "Напечатал" if ok else f"Не удалось напечатать: {out} (нужен Accessibility-доступ)"


@tool
def press_key(key: str) -> str:
    """Press a single named key in the active window: enter, tab, escape, pagedown, pageup, up, down, space.

    Args:
        key: Key name.
    """
    code = _KEYCODES.get(key.lower())
    if code is None:
        return f"Неизвестная клавиша '{key}'. Доступно: {', '.join(_KEYCODES)}"
    ok, out = _osa(f'tell application "System Events" to key code {code}')
    return f"Нажал {key}" if ok else f"Не удалось: {out}"


@tool
def send_telegram(text: str) -> str:
    """Write and SEND a message in the Telegram desktop app (must be open with a chat selected).
    Confirm with the user before sending anything sensitive.

    Args:
        text: Message text to type and send.
    """
    _run(["open", "-a", "Telegram"])
    ok, out = _osa(
        'delay 0.6\n'
        'tell application "System Events"\n'
        f'  keystroke "{_esc(text[:1500])}"\n'
        '  key code 36\n'  # Enter — отправить
        'end tell'
    )
    return "Отправил в Telegram" if ok else f"Не удалось отправить: {out} (Telegram открыт? Accessibility-доступ?)"
