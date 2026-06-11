"""Vision-слой: чтение фигур в PDF (рендер→vision). Vision замокан, без LLM/сети."""
import tempfile
from pathlib import Path

import pytest


def _make_pdf(text: str) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    p = Path(tempfile.mktemp(suffix=".pdf"))
    doc.save(str(p))
    doc.close()
    return p


def test_read_pdf_visual_renders_and_describes(monkeypatch):
    import src.media as M
    # мокаем vision → проверяем, что рендер страниц + вызов на каждую работает
    monkeypatch.setattr(M, "describe_image", lambda path, q="": "AXIS labels: egalitarian | authoritarian")
    pdf = _make_pdf("Figure with three axes")
    out = M.read_pdf_visual(pdf, "axis labels")
    assert "egalitarian" in out and "Стр. 1" in out


def test_pdf_vision_tool_builds():
    from src.media import make_pdf_vision_tool
    t = make_pdf_vision_tool()
    assert t.name == "read_pdf_figures"


def test_pdf_vision_tool_handles_bad_path():
    from src.media import make_pdf_vision_tool
    out = make_pdf_vision_tool().invoke({"path": "/nonexistent.pdf", "question": "x"})
    assert "не удалось" in out.lower() or "error" in out.lower()
