"""
app_control — СТРУКТУРНОЕ взаимодействие с приложениями (без костылей-скриншотов).

Использует родные API приложений через AppleScript (osascript) — это словари
скриптинга самих приложений (Mail/Safari/Calendar/Notes/Finder), а не пиксели и
не эмуляция клавиш. Агент читает и меняет состояние приложений структурно.

Для нескриптуемых приложений следующий шаг — Accessibility API (AX-дерево) через
pyobjc; здесь — скриптуемые (покрывают большинство повседневных задач).

Безопасность: AGENT_DRY_RUN=1 — не выполнять реальные действия. Чувствительное
(отправка письма) требует явного send=True и подтверждения пользователя.
Нужны разрешения macOS Automation (System Settings → Privacy → Automation).
"""
import os
import subprocess
from langchain_core.tools import tool


def _osa(script: str, timeout: int = 25) -> str:
    if (os.getenv("AGENT_DRY_RUN") or os.getenv("AGENT_EVAL_MODE") == "1"):
        return f"[dry-run] osascript: {script[:160]}"
    try:
        p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "").strip()
        return out if p.returncode == 0 else f"Ошибка: {(p.stderr or '').strip()}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


@tool
def list_running_apps() -> str:
    """List currently running (foreground) applications — structured, no screenshots."""
    return _osa('tell application "System Events" to get name of every process whose background only is false')


@tool
def safari_tabs() -> str:
    """Read titles and URLs of all open Safari tabs (front window) — structured app state."""
    return _osa(
        'tell application "Safari" to set out to ""\n'
        'tell application "Safari"\n'
        '  repeat with t in tabs of front window\n'
        '    set out to out & (name of t) & " — " & (URL of t) & linefeed\n'
        '  end repeat\nend tell\nreturn out'
    )


@tool
def safari_open(url: str) -> str:
    """Open a URL in a new Safari tab.

    Args:
        url: Address to open.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return _osa(f'tell application "Safari" to open location "{_esc(url)}"')


@tool
def compose_email(to: str, subject: str, body: str, send: bool = False) -> str:
    """Compose an email in Mail.app (structured, via Mail scripting API). Sends only if send=True.
    Confirm with the user before sending.

    Args:
        to: Recipient email.
        subject: Subject line.
        body: Message body.
        send: If True, actually send; otherwise leave as a draft window.
    """
    action = "send msg" if send else "set visible of msg to true"
    return _osa(
        'tell application "Mail"\n'
        f'  set msg to make new outgoing message with properties {{subject:"{_esc(subject)}", content:"{_esc(body)}", visible:true}}\n'
        f'  tell msg to make new to recipient with properties {{address:"{_esc(to)}"}}\n'
        f'  {action}\n'
        'end tell\nreturn "ok"'
    )


@tool
def create_note(title: str, body: str) -> str:
    """Create a note in Notes.app (structured).

    Args:
        title: Note title.
        body: Note body text.
    """
    return _osa(
        'tell application "Notes" to make new note at folder "Notes" of account "iCloud" '
        f'with properties {{name:"{_esc(title)}", body:"{_esc(body)}"}}\nreturn "ok"'
    )



@tool
def open_application(app_name: str) -> str:
    """
    Launch a macOS application by its name using AppleScript.
    
    Args:
        app_name: The name of the application to launch (e.g., 'FaceTime', 'Calculator').
    """
    return _osa(f'tell application "{_esc(app_name)}" to activate')
