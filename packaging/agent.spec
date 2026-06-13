# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — кроссплатформенная упаковка в ОДИН нативный бинарь
(Windows .exe / macOS / Linux). Сборка: `python scripts/build_binary.py`
или `pyinstaller packaging/agent.spec --clean --noconfirm`.

Подход onefile: config.yml кладётся в бандл и копируется рядом с бинарём при
первом запуске (см. main._frozen_bootstrap), данные (data/, config.local.yml)
персистятся в рабочей папке. Динамические импорты langchain/langgraph
собираются через collect_all (иначе PyInstaller их не находит).
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent  # корень репозитория (packaging/ → ..)

datas, binaries, hiddenimports = [], [], []

# Пакеты с динамическими/ленивыми импортами — собираем целиком.
for pkg in (
    "langchain", "langchain_core", "langchain_openai", "langchain_mcp_adapters",
    "langgraph", "langgraph_checkpoint", "langgraph_checkpoint_sqlite", "langgraph_prebuilt",
    "omegaconf", "trafilatura", "bm25s", "lightrag", "rich", "prompt_toolkit",
    "pydantic", "openai", "tiktoken",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:  # пакет может отсутствовать на части платформ
        print(f"[spec] collect_all({pkg}) пропущен: {e}")

# Наш код как ресурсы: дефолтный config.yml + навыки (читаются в рантайме).
datas += [
    (str(ROOT / "config.yml"), "."),
    (str(ROOT / "src" / "skills"), "src/skills"),
]
hiddenimports += collect_submodules("src")

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="self-extension-agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,          # CLI/REPL — нужна консоль
    disable_windowed_traceback=False,
    target_arch=None,      # нативная арх. раннера (CI собирает под каждую ОС)
    codesign_identity=None,
    entitlements_file=None,
)
