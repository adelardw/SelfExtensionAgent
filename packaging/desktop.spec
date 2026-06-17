# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — НАТИВНОЕ macOS-приложение `SEA.app` (десктоп: pywebview + FastAPI + React-GUI),
с Dock-иконкой (Ика + SEA, assets/sea_icon.icns). Windowed (без консоли), onedir → BUNDLE(.app).

Сборка (на macOS): `python scripts/build_binary.py --app`  ·  результат: dist/SEA.app
Перед сборкой собрать GUI: `cd frontend && npm install && npm run build`.

Отличие от agent.spec (CLI onefile-бинарь): entry = desktop.py, console=False, COLLECT+BUNDLE.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent

datas, binaries, hiddenimports = [], [], []

# Пакеты с динамическими импортами (LLM-стек + веб-сервер + нативное окно).
for pkg in (
    "langchain", "langchain_core", "langchain_openai", "langchain_mcp_adapters",
    "langgraph", "langgraph_checkpoint", "langgraph_checkpoint_sqlite", "langgraph_prebuilt",
    "omegaconf", "trafilatura", "bm25s", "lightrag", "rich", "prompt_toolkit",
    "pydantic", "openai", "tiktoken",
    "fastapi", "starlette", "uvicorn", "websockets", "webview",  # десктоп: сервер + нативное окно
    "objc",  # pyobjc-core — основа macOS-бэкенда pywebview
    "liteparse", "pymupdf", "openpyxl", "docx", "pptx",  # парсеры документов (PDF/xlsx/docx/pptx)
):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:  # noqa: BLE001
        print(f"[desktop.spec] collect_all({pkg}) пропущен: {e}")

# liteparse грузит libpdfium.dylib через dlopen (НЕ импорт) → collect_all мог не взять. Кладём ЯВНО
# рядом (в liteparse/), а PDFIUM_LIB_PATH на неё указывает рантайм (config_paths.bootstrap_frozen).
try:
    import liteparse as _lp
    _dyl = Path(_lp.__file__).parent / "libpdfium.dylib"
    if _dyl.exists():
        binaries += [(str(_dyl), "liteparse")]
except Exception as e:  # noqa: BLE001
    print(f"[desktop.spec] libpdfium.dylib не найдена: {e}")

# pywebview на macOS грузит бэкенд cocoa ДИНАМИЧЕСКИ (webview.platforms.cocoa → pyobjc-фреймворки).
# collect_all их не видит → бандл падал «pywebview не установлен». Тащим явно.
hiddenimports += collect_submodules("webview")
for _fw in ("Foundation", "AppKit", "WebKit", "Cocoa", "Quartz", "Security",
            "UniformTypeIdentifiers", "PyObjCTools"):
    try:
        hiddenimports += collect_submodules(_fw)
    except Exception:  # noqa: BLE001
        hiddenimports.append(_fw)

# Ресурсы: дефолтный config.yml, навыки, собранный GUI, иконки.
datas += [
    (str(ROOT / "config.yml"), "."),
    (str(ROOT / "src" / "skills"), "src/skills"),
]
_dist = ROOT / "frontend" / "dist"
if _dist.is_dir():
    datas += [(str(_dist), "frontend/dist")]
else:
    print("[desktop.spec] frontend/dist отсутствует — сначала `cd frontend && npm run build`")
for _ic in ("sea_icon.png", "sea_icon.icns", "sea_icon.ico"):
    if (ROOT / "assets" / _ic).exists():
        datas += [(str(ROOT / "assets" / _ic), "assets")]
hiddenimports += collect_submodules("src")

a = Analysis(
    [str(ROOT / "desktop.py")],
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
    pyz, a.scripts, [],
    exclude_binaries=True,         # onedir: бинарники едут в COLLECT (быстрый старт .app)
    name="SEA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                 # WINDOWED — нативное окно, без терминала
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "sea_icon.icns"),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, upx_exclude=[], name="SEA")

app = BUNDLE(
    coll,
    name="SEA.app",
    icon=str(ROOT / "assets" / "sea_icon.icns"),   # Dock-иконка (Ика + SEA)
    bundle_identifier="com.selfextension.sea",
    info_plist={
        "CFBundleName": "SEA",
        "CFBundleDisplayName": "SEA",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
        # Голосовой ввод: сервер пишет с мика через ffmpeg (avfoundation) → macOS попросит доступ.
        "NSMicrophoneUsageDescription": "SEA records your voice to transcribe it into the message box.",
    },
)
