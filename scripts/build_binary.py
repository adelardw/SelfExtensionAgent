#!/usr/bin/env python
"""
Кроссплатформенная сборка нативного бинаря через PyInstaller.

    uv sync --group package          # поставить pyinstaller
    python scripts/build_binary.py   # собрать под текущую ОС

Результат — один исполняемый файл в dist/ (Windows: .exe, macOS/Linux: бинарь).
Под Windows/macOS/Linux собирать НАДО на соответствующей ОС (PyInstaller не
кросс-компилирует) — для всех трёх сразу используется CI (.github/workflows/build.yml).
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "agent.spec"


def main() -> int:
    if not SPEC.exists():
        print(f"нет spec-файла: {SPEC}", file=sys.stderr)
        return 1
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller не установлен. Поставь: uv sync --group package", file=sys.stderr)
        return 1

    osname = platform.system()
    print(f"⚙  Сборка под {osname} ({platform.machine()})…")
    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC),
           "--clean", "--noconfirm",
           "--distpath", str(ROOT / "dist"),
           "--workpath", str(ROOT / "build")]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print("❌ сборка упала", file=sys.stderr)
        return rc

    exe = "self-extension-agent.exe" if osname == "Windows" else "self-extension-agent"
    out = ROOT / "dist" / exe
    print(f"✅ готово: {out}" if out.exists() else f"⚠ бинарь не найден на {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
