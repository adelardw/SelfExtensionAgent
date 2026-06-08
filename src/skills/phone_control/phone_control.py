"""
phone_control — структурное управление Android-телефоном через ADB (не скриншоты).

`uiautomator dump` отдаёт XML-ДЕРЕВО экрана (текст, resource-id, class, bounds,
clickable) — тот же принцип, что AX на десктопе. Агент читает элементы и тапает
ПО НИМ (по центру bounds), вводит текст, жмёт системные клавиши — структурно.

Требуется `adb` (brew install android-platform-tools) + подключённое устройство с
включённой отладкой по USB. AGENT_DRY_RUN=1 — действия не выполняются (чтение ок).
"""
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from langchain_core.tools import tool

_BOUNDS = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")
_KEYS = {"back": "KEYCODE_BACK", "home": "KEYCODE_HOME", "enter": "KEYCODE_ENTER",
         "tab": "KEYCODE_TAB", "menu": "KEYCODE_MENU", "power": "KEYCODE_POWER"}


def _adb(args: list[str], timeout: int = 25) -> tuple[bool, str]:
    try:
        p = subprocess.run(["adb", *args], capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout or p.stderr).strip()
    except FileNotFoundError:
        return False, "adb не установлен. Поставь: brew install android-platform-tools"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _device_ready() -> tuple[bool, str]:
    ok, out = _adb(["devices"])
    if not ok:
        return False, out
    devs = [l for l in out.splitlines()[1:] if l.strip().endswith("\tdevice")]
    if not devs:
        return False, "Нет подключённого устройства (включи отладку по USB и подтверди на телефоне). `adb devices` пуст."
    return True, devs[0].split("\t")[0]


def _center(bounds: str):
    m = _BOUNDS.search(bounds or "")
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return (x1 + x2) // 2, (y1 + y2) // 2


def _dump_tree() -> tuple[bool, str]:
    ok, _ = _adb(["shell", "uiautomator", "dump", "/sdcard/agent_dump.xml"])
    if not ok:
        return False, "Не удалось снять дамп UI."
    return _adb(["shell", "cat", "/sdcard/agent_dump.xml"])


@tool
def phone_ui() -> str:
    """Read the current Android screen as a structured element tree (text/id/clickable) — not a screenshot."""
    ready, info = _device_ready()
    if not ready:
        return info
    ok, xml = _dump_tree()
    if not ok:
        return xml
    try:
        root = ET.fromstring(xml)
    except Exception as e:  # noqa: BLE001
        return f"Не распарсил дамп: {e}"
    lines = []
    for n in root.iter("node"):
        txt = n.get("text") or n.get("content-desc") or ""
        rid = (n.get("resource-id") or "").split("/")[-1]
        if not (txt or (rid and n.get("clickable") == "true")):
            continue
        c = _center(n.get("bounds"))
        tag = "🔘" if n.get("clickable") == "true" else "·"
        lines.append(f"{tag} {txt[:40] or rid}  [{rid}] @{c}")
        if len(lines) >= 60:
            break
    return "Экран Android (структурно):\n" + ("\n".join(lines) or "(пусто)")


@tool
def phone_tap(query: str) -> str:
    """Tap the screen element whose text or resource-id matches `query` (by element, not pixels).

    Args:
        query: Visible text or resource-id of the element to tap.
    """
    if os.getenv("AGENT_DRY_RUN"):
        return f"[dry-run] tap '{query}'"
    ready, info = _device_ready()
    if not ready:
        return info
    ok, xml = _dump_tree()
    if not ok:
        return xml
    root = ET.fromstring(xml)
    for n in root.iter("node"):
        hay = f"{n.get('text','')} {n.get('content-desc','')} {n.get('resource-id','')}".lower()
        if query.lower() in hay:
            c = _center(n.get("bounds"))
            if c:
                ok2, out = _adb(["shell", "input", "tap", str(c[0]), str(c[1])])
                return f"Тапнул '{query}' @{c}" if ok2 else f"Не удалось: {out}"
    return f"Элемент '{query}' не найден на экране."


@tool
def phone_type(text: str) -> str:
    """Type text into the focused field on the phone.

    Args:
        text: Text to input.
    """
    if os.getenv("AGENT_DRY_RUN"):
        return f"[dry-run] type '{text[:40]}'"
    ready, info = _device_ready()
    if not ready:
        return info
    ok, out = _adb(["shell", "input", "text", text.replace(" ", "%s")])
    return "Ввёл текст" if ok else f"Ошибка: {out}"


@tool
def phone_key(key: str) -> str:
    """Press a system key: back, home, enter, tab, menu, power.

    Args:
        key: Key name.
    """
    code = _KEYS.get(key.lower())
    if not code:
        return f"Неизвестная клавиша. Доступно: {', '.join(_KEYS)}"
    if os.getenv("AGENT_DRY_RUN"):
        return f"[dry-run] key {key}"
    ok, out = _adb(["shell", "input", "keyevent", code])
    return f"Нажал {key}" if ok else f"Ошибка: {out}"


@tool
def phone_open_app(package: str) -> str:
    """Launch an Android app by package name (e.g. 'org.telegram.messenger').

    Args:
        package: Android package id.
    """
    if os.getenv("AGENT_DRY_RUN"):
        return f"[dry-run] open app {package}"
    ready, info = _device_ready()
    if not ready:
        return info
    ok, out = _adb(["shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"])
    return f"Запустил {package}" if ok else f"Ошибка: {out}"


@tool
def phone_apps(filter: str = "") -> str:
    """List installed app packages (optionally filtered).

    Args:
        filter: Substring to filter package names.
    """
    ready, info = _device_ready()
    if not ready:
        return info
    ok, out = _adb(["shell", "pm", "list", "packages"])
    if not ok:
        return out
    pkgs = [l.replace("package:", "") for l in out.splitlines() if (not filter or filter.lower() in l.lower())]
    return "\n".join(pkgs[:60]) or "Ничего не найдено."
