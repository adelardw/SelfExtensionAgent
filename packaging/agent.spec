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

# Наш код как ресурсы: дефолтный config.yml + навыки + собранный GUI (frontend/dist).
# GUI собирается до упаковки: `cd frontend && npm install && npm run build`.
datas += [
    (str(ROOT / "config.yml"), "."),
    (str(ROOT / "src" / "skills"), "src/skills"),
]
# Иконка приложения (Ика + SEA) — в бандл (фавикон GUI, окно десктопа).
_icons = ROOT / "assets"
for _ic in ("sea_icon.png", "sea_icon.icns", "sea_icon.ico"):
    if (_icons / _ic).exists():
        datas += [(str(_icons / _ic), "assets")]
_dist = ROOT / "frontend" / "dist"
if _dist.is_dir():
    datas += [(str(_dist), "frontend/dist")]
else:
    print("[spec] frontend/dist отсутствует — GUI не попадёт в бинарь (сначала npm run build)")
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
    # Иконка: Windows читает .ico, macOS .app — .icns (для onefile-консоли ОС может игнорировать,
    # но при сборке .app/через CI применится). Список — PyInstaller берёт подходящий под платформу.
    icon=[str(ROOT / "assets" / "sea_icon.ico"), str(ROOT / "assets" / "sea_icon.icns")],
)
