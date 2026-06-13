"""
ax_control — структурное управление ЛЮБЫМИ приложениями через Accessibility API (AX).

Без костылей: ОС отдаёт ДЕРЕВО UI-элементов (роль/заголовок/значение) — как у
скринридеров. Агент читает состояние приложения и действует ПО ЭЛЕМЕНТАМ (нажать
кнопку, заполнить поле), а НЕ по пикселям/скриншотам. Покрывает нескриптуемые
приложения (где AppleScript не помогает).

Требуется разрешение macOS Accessibility (System Settings → Privacy → Accessibility:
добавить терминал/python). AGENT_DRY_RUN=1 — не выполнять действия (чтение безопасно).
"""
import os
from langchain_core.tools import tool

try:
    import AppKit
    from ApplicationServices import (
        AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
        AXUIElementPerformAction, AXUIElementSetAttributeValue, AXIsProcessTrusted,
        kAXChildrenAttribute, kAXRoleAttribute, kAXTitleAttribute,
        kAXValueAttribute, kAXDescriptionAttribute, kAXPressAction,
    )
    _AX = True
except Exception:  # noqa: BLE001
    _AX = False


def _pid(app_name: str):
    ws = AppKit.NSWorkspace.sharedWorkspace()
    for a in ws.runningApplications():
        if (a.localizedName() or "").lower() == app_name.lower():
            return a.processIdentifier()
    return None


def _attr(elem, attr):
    err, val = AXUIElementCopyAttributeValue(elem, attr, None)
    return val if err == 0 else None


def _label(elem) -> str:
    for a in (kAXTitleAttribute, kAXValueAttribute, kAXDescriptionAttribute):
        v = _attr(elem, a)
        if v:
            return str(v)
    return ""


def _children(elem) -> list:
    v = _attr(elem, kAXChildrenAttribute)
    return list(v) if v else []


def _walk(elem, depth, out, max_depth, max_nodes):
    if len(out) >= max_nodes or depth > max_depth:
        return
    role = _attr(elem, kAXRoleAttribute)
    if role:
        lbl = _label(elem)
        out.append("  " * depth + (f"{role}: {lbl[:60]}" if lbl else str(role)))
    for c in _children(elem):
        _walk(c, depth + 1, out, max_depth, max_nodes)


def _find(elem, label, depth=0, max_depth=8):
    if depth > max_depth:
        return None
    lbl = _label(elem)
    if lbl and label.lower() in lbl.lower():
        return elem
    for c in _children(elem):
        r = _find(c, label, depth + 1, max_depth)
        if r is not None:
            return r
    return None


def _guard(app_name: str):
    if not _AX:
        return None, "Accessibility API недоступен (pyobjc не установлен / не macOS)."
    if not AXIsProcessTrusted():
        return None, ("Нет доступа Accessibility. Включи: System Settings → Privacy & Security → "
                      "Accessibility → добавь Terminal/Python.")
    pid = _pid(app_name)
    if not pid:
        return None, f"Приложение '{app_name}' не запущено."
    return AXUIElementCreateApplication(pid), ""


@tool
def read_ui(app_name: str, max_depth: int = 4) -> str:
    """Read the structured UI element tree of an app (roles/titles/values) — NOT a screenshot.

    Args:
        app_name: Application name as in macOS (e.g. 'Notes', 'System Settings').
        max_depth: Tree depth to traverse.
    """
    app, err = _guard(app_name)
    if err:
        return err
    out: list[str] = []
    _walk(app, 0, out, max_depth, 80)
    return f"UI-дерево {app_name} (структурно):\n" + ("\n".join(out) or "(пусто)")


@tool
def click_element(app_name: str, label: str) -> str:
    """Press a UI element (button/menu/etc.) found by its visible label — by element, not pixels.

    Args:
        app_name: Application name.
        label: Visible text/label of the element to press.
    """
    if os.getenv("AGENT_DRY_RUN") or os.getenv("AGENT_EVAL_MODE") == "1":
        return f"[dry-run] press '{label}' in {app_name}"
    app, err = _guard(app_name)
    if err:
        return err
    el = _find(app, label)
    if el is None:
        return f"Элемент '{label}' не найден в {app_name}."
    e = AXUIElementPerformAction(el, kAXPressAction)
    return f"Нажал '{label}'" if e == 0 else f"Не удалось нажать '{label}' (код {e})."


@tool
def set_field(app_name: str, label: str, text: str) -> str:
    """Set the value of a text field found by its label — structurally.

    Args:
        app_name: Application name.
        label: Label/placeholder of the field.
        text: Text to put into the field.
    """
    if os.getenv("AGENT_DRY_RUN") or os.getenv("AGENT_EVAL_MODE") == "1":
        return f"[dry-run] set '{label}' = '{text[:40]}' in {app_name}"
    app, err = _guard(app_name)
    if err:
        return err
    el = _find(app, label)
    if el is None:
        return f"Поле '{label}' не найдено в {app_name}."
    e = AXUIElementSetAttributeValue(el, kAXValueAttribute, text)
    return f"Заполнил '{label}'" if e == 0 else f"Не удалось заполнить '{label}' (код {e})."
