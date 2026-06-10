"""Медиа-вложения (без LLM-путей) + прогресс-вью + SearXNG cooldown."""
import time

from src.media import AUDIO_EXTS, IMAGE_EXTS, TEXT_EXTS, attachment_context
from src.progress import ProgressView


def test_attachment_context_text_inline(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("важная заметка про бюджет", encoding="utf-8")
    ctx = attachment_context([str(f)])
    assert "важная заметка" in ctx and "notes.txt" in ctx


def test_attachment_context_truncates_long_text(tmp_path):
    f = tmp_path / "big.log"
    f.write_text("x" * 20000, encoding="utf-8")
    ctx = attachment_context([str(f)])
    assert "обрезано" in ctx and len(ctx) < 8000


def test_attachment_context_binary_and_missing(tmp_path):
    b = tmp_path / "data.sqlite"
    b.write_bytes(b"\x00\x01\x02")
    ctx = attachment_context([str(b), str(tmp_path / "nope.txt")])
    assert "бинарный/другой формат" in ctx
    assert "не найден" in ctx


def test_ext_sets_disjoint():
    assert not (IMAGE_EXTS & TEXT_EXTS) and not (IMAGE_EXTS & AUDIO_EXTS)


def test_progress_view_flow():
    pv = ProgressView()
    assert "режим" in pv.on_update("reflexion", {"mode": "heavy"})
    assert "3 шаг" in pv.on_update("decompose", {"subtasks": [
        {"goal": "a"}, {"goal": "b"}, {"goal": "c"}]})
    lbl = pv.on_update("step_executor", {"current_step": 1})
    assert "1/3" in lbl and "a" in lbl
    assert "ретрай" in pv.on_update("step_executor", {"step_retries": 1})
    # ревью добавил подшаги доработки
    lbl = pv.on_update("review", {"subtasks": [{}] * 5})
    assert "+2" in lbl
    assert pv.on_update("validation", {}) is not None
    assert pv.on_update("unknown_node", {}) is None


def test_missing_module_detection():
    from src.utils import MODULE_TO_PKG, ensure_python_package, missing_module_from_error

    assert missing_module_from_error("ModuleNotFoundError: No module named 'pptx'") == "pptx"
    assert missing_module_from_error("No module named 'bs4.element'") == "bs4"
    assert missing_module_from_error("ValueError: что-то другое") == ""
    assert MODULE_TO_PKG["pptx"] == "python-pptx"
    # уже установленный модуль → ok без установки
    ok, note = ensure_python_package("json")
    assert ok and "уже установлен" in note


def test_searxng_cooldown(monkeypatch):
    import importlib

    from src.skills.web_search import web_search as ws

    monkeypatch.setattr(ws, "_SEARXNG", "http://localhost:1")  # закрытый порт
    monkeypatch.delenv("SEARXNG_DOWN_UNTIL", raising=False)
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise OSError("Connection refused")

    monkeypatch.setattr(ws, "_search_searxng", _boom)
    monkeypatch.setattr(ws, "_CLOAK", False)
    monkeypatch.setattr(ws, "_search_fallback", lambda *a, **k: [{"title": "t", "url": "u", "snippet": "s"}])

    ws.search_web.invoke({"query": "тест"})
    assert calls["n"] == 1 and ws._searxng_down_until() > time.time()
    # второй вызов в окне cooldown НЕ дёргает SearXNG (нет спама)
    ws.search_web.invoke({"query": "тест2"})
    assert calls["n"] == 1

    # cooldown ПЕРЕЖИВАЕТ перезагрузку модуля (навыки reload'ятся на каждом шаге)
    ws2 = importlib.reload(ws)
    assert ws2._searxng_down_until() > time.time()
