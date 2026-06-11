"""Трек A: контекстный инжиниринг поиска — chunk → BM25S → (vector) → budget. Офлайн."""
import os

from src.skills.web_search.web_search import _bm25_top, _chunk, _relevant_chunks


def test_chunking_splits_long_text():
    text = "\n\n".join(f"Абзац {i} с каким-то содержанием про разные темы." for i in range(10))
    chunks = _chunk(text)
    assert len(chunks) >= 5 and all(isinstance(c, str) for c in chunks)


def test_bm25_filters_to_relevant_only():
    chunks = ["погода и котики"] * 5 + ["Выручка Acme за 2023 составила 4.7 млрд долларов"]
    idx = _bm25_top(chunks, "выручка Acme 2023 доллары", top=20)
    assert idx[0] == 5          # релевантный чанк — первый
    assert 0 not in idx[:1]     # шум не лезет вперёд


def test_relevant_chunks_returns_only_fact(monkeypatch):
    # без ключа эмбеддингов → BM25S-only ветка (детерминированно, без сети)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    noise = "\n\n".join(f"Абзац {i} про погоду и котиков, не по делу." for i in range(30))
    text = noise + "\n\nВыручка Acme за 2023 год составила 4.7 миллиарда долларов.\n\n" + noise
    out = _relevant_chunks(text, "выручка Acme 2023 доллары", budget=300)
    assert "4.7" in out and "Acme" in out      # факт найден
    assert "погоду" not in out                  # шум отфильтрован
    assert len(out) <= 360                       # в пределах budget


def test_empty_find_returns_head():
    text = "x" * 500
    assert len(_relevant_chunks(text, "", budget=100)) == 100
