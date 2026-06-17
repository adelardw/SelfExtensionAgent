"""Картинки в чате: image-поиск отдаёт markdown, anti-URL фильтр их не вырезает."""
import json

import src.agent as A
import src.tools.image_search as IS


def test_strip_ungrounded_preserves_images_even_if_cdn_not_grounded():
    # CDN картинки почти всегда ≠ домен-источник → без исключения фильтр бы её срезал.
    ans = ("![кот](https://cdn.imghost.org/c.jpg)\n"
           "[пруф](https://realsource.ru/a) и выдумка https://hallucinated.ru/x")
    out = A._strip_ungrounded_urls(ans, grounded={"realsource.ru"})
    assert "![кот](https://cdn.imghost.org/c.jpg)" in out   # картинка цела
    assert "https://realsource.ru/a" in out                # заземлённая ссылка цела
    assert "hallucinated.ru" not in out                    # выдуманный URL вырезан


def test_strip_ungrounded_preserves_gallery_block():
    # Блок-галерея целиком вне фильтра (URL картинок из реального поиска, домены не «заземлены»).
    gal = ("```sea-gallery\n# панда\n"
           "https://cdn.x.org/1.jpg ||| Панда ||| https://src.io/p\n```")
    out = A._strip_ungrounded_urls("Вот фото:\n\n" + gal + "\n\nКрасиво.", grounded=set())
    assert "https://cdn.x.org/1.jpg" in out      # img URL цел
    assert "https://src.io/p" in out             # источник цел
    assert out.startswith("Вот фото") and out.rstrip().endswith("Красиво.")


def test_search_images_parses_searxng(monkeypatch):
    payload = {"results": [
        {"img_src": "//up.wmedia.org/a.jpg", "title": "Panda", "url": "https://en.wiki/Panda"},
        {"thumbnail_src": "https://t.io/b.jpg", "title": "", "url": "https://x.io"},
    ]}

    class _R:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
    monkeypatch.setattr(IS.urllib.request, "urlopen", lambda *a, **k: _R())
    md = IS.search_images("красная панда", 4)
    assert "```sea-gallery" in md and md.rstrip().endswith("```")     # структурный блок-галерея
    assert "https://up.wmedia.org/a.jpg ||| Panda |||" in md          # // → https:// нормализовано
    assert "https://t.io/b.jpg ||| красная панда |||" in md          # пустой title → запрос как подпись


def test_search_images_graceful_when_no_backend(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)  # нет SearXNG

    def _boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(IS.urllib.request, "urlopen", _boom)  # DDG-фолбэк тоже лёг
    out = IS.search_images("что угодно")
    assert "недоступен" in out  # не падает, а честно сообщает
