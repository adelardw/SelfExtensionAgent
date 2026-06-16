"""
device_control — действия с устройством по запросу из чата (on-demand, не мониторинг).

Принцип (как просил пользователь): агент НЕ анализирует в фоне, что открыто.
Только при обращении в чат он может открыть сайт/приложение, сделать скриншот
(и «посмотреть» на него vision-моделью), показать уведомление или озвучить ответ.

КРОССПЛАТФОРМЕННО: базовые операции (открыть/скриншот/уведомление/TTS) имеют
бэкенды под macOS / Linux / Windows и выбираются по platform.system() — штатными
утилитами, без сторонних зависимостей. Где утилиты нет — навык деградирует с
подсказкой, что доставить. Работа с УЖЕ ОТКРЫТЫМИ окнами (keystroke/scroll/AX) —
пока только macOS (System Events); на других ОС честно сообщает об этом.
"""
import os
import platform
import shutil
import subprocess
import tempfile
import time
from langchain_core.tools import tool

_OS = platform.system()  # 'Darwin' | 'Linux' | 'Windows'


def _run(cmd: list[str], timeout: int = 20) -> tuple[bool, str]:
    # Безопасный режим: AGENT_DRY_RUN=1 ИЛИ AGENT_EVAL_MODE=1 (бенч/eval) → НЕ выполнять реальные
    # действия с устройством. Критично: иначе бенч-прогон дёргал бы `open <url>`/keystroke в ЖИВОЙ
    # системе юзера — крал фокус, печатал в его активную вкладку (живой инцидент).
    if os.getenv("AGENT_DRY_RUN") or os.getenv("AGENT_EVAL_MODE") == "1":
        return True, f"[dry-run] {' '.join(str(c) for c in cmd)[:200]}"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _first_tool(*names: str) -> str:
    """Первый доступный в PATH бинарник из списка (для Linux-бэкендов)."""
    for n in names:
        if shutil.which(n):
            return n
    return ""


@tool
def open_url(url: str) -> str:
    """Open a URL or web page in the default browser (on-demand). Cross-platform.

    Args:
        url: Address to open (http/https).
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if _OS == "Darwin":
        # -g: открыть В ФОНЕ, не выдёргивая пользователя из текущего окна (физ-браузер —
        # тихий бонус, без кражи фокуса). Воспроизведение идёт отдельным путём (browser_bridge).
        ok, out = _run(["open", "-g", url])
    elif _OS == "Windows":
        ok, out = _run(["cmd", "/c", "start", "", url])
    else:  # Linux/BSD
        opener = _first_tool("xdg-open", "gio", "sensible-browser")
        if not opener:
            return "Нет открывалки URL. Установи: sudo apt install xdg-utils"
        ok, out = _run([opener, url] if opener != "gio" else ["gio", "open", url])
    return f"Открыл {url}" if ok else f"Не удалось открыть: {out}"


@tool
def open_app(app_name: str) -> str:
    """Open a desktop application by name (e.g. 'Mail', 'Safari', 'firefox'). Cross-platform.

    Args:
        app_name: Application name (macOS app name / Linux binary or .desktop / Windows app).
    """
    if _OS == "Darwin":
        ok, out = _run(["open", "-a", app_name])
    elif _OS == "Windows":
        ok, out = _run(["cmd", "/c", "start", "", app_name])
    else:  # Linux: gtk-launch по .desktop, иначе прямой запуск бинарника
        if shutil.which("gtk-launch") and _run(["gtk-launch", app_name]) [0]:
            return f"Открыл приложение {app_name}"
        if shutil.which(app_name):
            try:
                if not os.getenv("AGENT_DRY_RUN"):
                    subprocess.Popen([app_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Открыл приложение {app_name}"
            except Exception as e:  # noqa: BLE001
                return f"Не удалось открыть {app_name}: {e}"
        return f"Приложение '{app_name}' не найдено (нет в PATH и gtk-launch не сработал)."
    return f"Открыл приложение {app_name}" if ok else f"Не удалось открыть {app_name}: {out}"


@tool
def capture_screen() -> str:
    """Take a screenshot so the agent can 'look at' the screen. Cross-platform.
    Returns the saved image path. For automatic analysis use analyze_screen instead.
    """
    path = f"{tempfile.gettempdir()}/agent_screen_{int(time.time())}.png"
    if _OS == "Darwin":
        ok, out = _run(["screencapture", "-x", path])
    elif _OS == "Windows":
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
            "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen; "
            "$bmp=New-Object Drawing.Bitmap $b.Width,$b.Height; "
            "$g=[Drawing.Graphics]::FromImage($bmp); "
            "$g.CopyFromScreen($b.Location,[Drawing.Point]::Empty,$b.Size); "
            f"$bmp.Save('{path}')"
        )
        ok, out = _run(["powershell", "-NoProfile", "-Command", ps], timeout=30)
    else:  # Linux: grim (wayland) / scrot / imagemagick import / maim
        tool_name = _first_tool("grim", "scrot", "maim", "import")
        if not tool_name:
            return "Нет утилиты скриншота. Установи одну из: grim | scrot | maim | imagemagick"
        cmd = {"grim": [tool_name, path], "scrot": [tool_name, path],
               "maim": [tool_name, path], "import": [tool_name, "-window", "root", path]}[tool_name]
        ok, out = _run(cmd, timeout=30)
    if os.getenv("AGENT_DRY_RUN"):
        return f"[dry-run] скриншот → {path}"
    return f"Скриншот сохранён: {path}" if ok else f"Не удалось снять экран: {out}"


@tool
def analyze_screen(question: str = "") -> str:
    """Take a screenshot AND analyze it with the vision model in one step (FALLBACK eyes).

    Prefer STRUCTURED access first — it is exact and cheap: scriptable apps via
    app_control (native APIs), other apps via ax_control (read_ui/click_element).
    Use analyze_screen ONLY when structured access is unavailable: the app exposes
    neither a script API nor an accessibility tree; the content is inherently visual
    (chart/diagram/photo/video/rendered PDF); the platform is not macOS; or you need
    to visually CONFIRM a result. This is a costly multimodal call, imprecise on
    coordinates — do not make it your default.

    Args:
        question: Optional focus — what to pay attention to on the screen.
    """
    res = capture_screen.invoke({})
    if "сохранён" not in res and "dry-run" not in res:
        return res  # ошибка снятия экрана
    path = res.split(": ", 1)[-1].strip() if ": " in res else ""
    if os.getenv("AGENT_DRY_RUN"):
        return f"[dry-run] vision-анализ экрана ({path})"
    try:
        from src.media import describe_image

        return "Вижу на экране:\n" + describe_image(path, question)
    except Exception as e:  # noqa: BLE001
        return f"Скриншот снят ({path}), но vision-анализ не сработал: {e}"


@tool
def notify(title: str, message: str) -> str:
    """Show a desktop notification. Cross-platform.

    Args:
        title: Notification title.
        message: Notification body.
    """
    # message/title — LLM-управляемы (steerable инъекцией) → ЭКРАНИРУЕМ: notify в _DEFAULT_READONLY,
    # HITL не зовётся никогда, так что неэкранированная интерполяция = injection→RCE без чекпойнта
    # (баг ревью RESIDUAL-B, острее SEC-1). osascript → _esc, PowerShell → _ps_esc.
    if _OS == "Darwin":
        ok, out = _run(["osascript", "-e",
                        f'display notification "{_esc(message)}" with title "{_esc(title)}"'])
    elif _OS == "Windows":
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$n=New-Object System.Windows.Forms.NotifyIcon; "
            "$n.Icon=[System.Drawing.SystemIcons]::Information; $n.Visible=$true; "
            f"$n.ShowBalloonTip(5000,'{_ps_esc(title)}','{_ps_esc(message)}',[System.Windows.Forms.ToolTipIcon]::Info)"
        )
        ok, out = _run(["powershell", "-NoProfile", "-Command", ps])
    else:  # Linux
        if not shutil.which("notify-send"):
            return "Нет notify-send. Установи: sudo apt install libnotify-bin"
        ok, out = _run(["notify-send", title, message])
    return "Уведомление показано" if ok else f"Не удалось уведомить: {out}"


@tool
def speak(text: str) -> str:
    """Speak text aloud via the system voice (free local TTS). Cross-platform.

    Args:
        text: Text to read aloud.
    """
    t = text[:500]
    if _OS == "Darwin":
        ok, out = _run(["say", t])
    elif _OS == "Windows":
        ps = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{_ps_esc(t)}')"
        ok, out = _run(["powershell", "-NoProfile", "-Command", ps], timeout=30)
    else:  # Linux
        tts = _first_tool("spd-say", "espeak-ng", "espeak")
        if not tts:
            return "Нет TTS. Установи: sudo apt install speech-dispatcher (spd-say) или espeak-ng"
        ok, out = _run([tts, t], timeout=30)
    return "Озвучил" if ok else f"Не удалось озвучить: {out}"


# ── работа с УЖЕ ОТКРЫТЫМИ приложениями (System Events; нужен Accessibility-доступ) ──

_KEYCODES = {"enter": 36, "return": 36, "tab": 48, "escape": 53, "esc": 53,
             "pagedown": 121, "pageup": 116, "down": 125, "up": 126, "space": 49, "home": 115, "end": 119}


_MACOS_ONLY = "Эта операция (работа с активным окном) пока только на macOS. " \
              "Кроссплатформенные действия: open_url/open_app/capture_screen/analyze_screen/notify/speak."


def _osa(script: str) -> tuple[bool, str]:
    if _OS != "Darwin":
        return False, _MACOS_ONLY
    return _run(["osascript", "-e", script])


def _esc(text: str) -> str:
    """Экранирование для строкового литерала AppleScript (двойные кавычки)."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _ps_esc(text: str) -> str:
    """Экранирование для одинарно-кавыченного литерала PowerShell: ' → '' (удвоение). Внутри
    '...' PowerShell спецсимволы (` $) не интерпретируются, выйти можно ТОЛЬКО незакрытой '."""
    return text.replace("'", "''")


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
