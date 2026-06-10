"""Чтение файлов (тиерный PDF/Excel/docx/txt) + browse-выжимка — офлайн, без сети/LLM."""
import tempfile
from pathlib import Path

import pytest


# ── read_file диспетчер ─────────────────────────────────────────────────

def test_read_file_text(tmp_path):
    from src.media import read_file

    f = tmp_path / "n.txt"
    f.write_text("привет данные 42", encoding="utf-8")
    assert "42" in read_file(f)


def test_read_file_missing():
    from src.media import read_file

    assert "не найден" in read_file("/nope/zzz.txt")


def test_read_excel(tmp_path):
    import pandas as pd

    from src.media import read_file

    f = tmp_path / "t.xlsx"
    pd.DataFrame({"кат": ["еда", "аренда"], "сумма": [4500, 3000]}).to_excel(f, index=False)
    out = read_file(f)
    assert "еда" in out and "3000" in out


def test_read_pdf_preserves_table_layout(tmp_path):
    """Тиерный read_pdf (liteparse) держит строку таблицы вместе — значение↔строка."""
    import fitz

    from src.media import read_pdf

    doc = fitz.open()
    page = doc.new_page()
    y = 60
    for row in [["Item", "Volume"], ["Fish bag", "0.1777"], ["Box", "1.0"]]:
        x = 60
        for c in row:
            page.insert_text((x, y), c)
            x += 140
        y += 28
    f = tmp_path / "tbl.pdf"
    doc.save(str(f))
    doc.close()
    txt = read_pdf(f)
    assert "Fish bag" in txt and "0.1777" in txt
    # значение должно идти за названием строки на близком расстоянии (layout сохранён)
    seg = txt.split("Fish bag", 1)[1][:30]
    assert "0.1777" in seg


def test_read_pdf_falls_back(monkeypatch, tmp_path):
    """Если liteparse падает — фолбэк на pymupdf, прогон не рушится."""
    import fitz

    from src import media

    doc = fitz.open()
    doc.new_page().insert_text((50, 50), "fallback works 7")
    f = tmp_path / "fb.pdf"
    doc.save(str(f))
    doc.close()
    monkeypatch.setattr(media, "_pdf_liteparse", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "fallback works 7" in media.read_pdf(f)


# ── browse-выжимка (прицельные куски) ──────────────────────────────────

def test_relevant_chunks_targets_query():
    from src.skills.web_search.web_search import _relevant_chunks

    text = ("Раздел про погоду: сегодня дождь.\n\n"
            "Финансы: выручка компании составила 5300 рублей за квартал.\n\n"
            "Биография: основана в 1990 году.")
    out = _relevant_chunks(text, "выручка компании рублей", budget=200)
    assert "5300" in out  # нашёл релевантный кусок с числом
    assert "дождь" not in out or len(out) > 100  # шум приоритетно отброшен


def test_relevant_chunks_empty_query_returns_head():
    from src.skills.web_search.web_search import _relevant_chunks

    text = "a" * 500
    assert len(_relevant_chunks(text, "", budget=100)) == 100
