#!/usr/bin/env python
"""
Кроссплатформенная сборка нативного бинаря через PyInstaller.

    uv sync --group package          # поставить pyinstaller
    python scripts/build_binary.py        # CLI-бинарь под текущую ОС (onefile)
    python scripts/build_binary.py --app  # macOS .app десктоп (Dock-иконка Ика+SEA)

CLI → один исполняемый файл в dist/ (Windows: .exe, macOS/Linux: бинарь).
--app → dest/SEA.app (только macOS): нативное окно (pywebview + FastAPI + React-GUI), Dock-иконка.
Перед --app собрать GUI: `cd frontend && npm install && npm run build`.
Под каждую ОС собирать НА соответствующей ОС (PyInstaller не кросс-компилирует) — CI: .github/workflows/build.yml.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "agent.spec"
DESKTOP_SPEC = ROOT / "packaging" / "desktop.spec"


def main() -> int:
    app_mode = "--app" in sys.argv[1:] or "--desktop" in sys.argv[1:]
    osname = platform.system()

    if app_mode and osname != "Darwin":
        print("--app (нативный .app) собирается только на macOS. Для других ОС — обычная сборка.",
              file=sys.stderr)
        return 2

    spec = DESKTOP_SPEC if app_mode else SPEC
    if not spec.exists():
        print(f"нет spec-файла: {spec}", file=sys.stderr)
        return 1
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller не установлен. Поставь: uv sync --group package", file=sys.stderr)
        return 1
    if app_mode and not (ROOT / "frontend" / "dist").is_dir():
        print("⚠ frontend/dist не собран — GUI будет пустым. Сначала: cd frontend && npm run build",
              file=sys.stderr)

    target = "SEA.app (десктоп)" if app_mode else "CLI-бинарь"
    print(f"⚙  Сборка {target} под {osname} ({platform.machine()})…")
    cmd = [sys.executable, "-m", "PyInstaller", str(spec),
           "--clean", "--noconfirm",
           "--distpath", str(ROOT / "dist"),
           "--workpath", str(ROOT / "build")]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print("❌ сборка упала", file=sys.stderr)
        return rc

    if app_mode:
        out = ROOT / "dist" / "SEA.app"
        print(f"✅ готово: {out}  (Dock-иконка Ика+SEA)" if out.exists()
              else f"⚠ .app не найден на {out}")
    else:
        exe = "self-extension-agent.exe" if osname == "Windows" else "self-extension-agent"
        out = ROOT / "dist" / exe
        print(f"✅ готово: {out}" if out.exists() else f"⚠ бинарь не найден на {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
