"""Мультиселект в REPL-уточнениях: отметить несколько вариантов («1,3») + дописать своё."""
import os

import pytest

needs_key = pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="main импортирует src.agent (LLM строится на импорте)",
)


@needs_key
def test_resolve_choice_multiselect():
    from main import _resolve_choice
    opts = ["Яндекс Музыка", "Spotify", "YouTube Music"]
    # несколько номеров → несколько вариантов
    assert _resolve_choice("1,3", opts) == "Яндекс Музыка; YouTube Music"
    # номер + дописанный свой текст
    assert _resolve_choice("1 и плюс ещё вот это", opts) == "Яндекс Музыка; и плюс ещё вот это"
    # одиночный номер — как раньше
    assert _resolve_choice("2", opts) == "Spotify"
    # «сам реши» → допущение (пусто)
    assert _resolve_choice("сам реши", opts) == ""
    # свободный текст без цифр — как есть
    assert _resolve_choice("какой-то свой ответ", opts) == "какой-то свой ответ"
    # пусто → допущение
    assert _resolve_choice("", opts) == ""
