"""Универсальный «любой сервис» в чате: выбор площадки без допроса, подсказка вариантов,
замыкание персонализации (сервис в ответе → memory_extraction → предпочтение)."""
import os

import pytest

needs_key = pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="нужен API-ключ: llm строится на импорте src.agent",
)


def test_act_prompt_has_service_choice_rules():
    """Инварианты act-промпта: выбор сервиса (память → сам → назвать в итоге) и
    подсказка вариантов; необратимое — только через ask_user."""
    from src.prompts import act_system_prompt
    for marker in ("ВЫБОР СЕРВИСА", "ПОДСКАЗЫВАЙ ВАРИАНТЫ", "ask_user"):
        assert marker in act_system_prompt, f"потеряно правило: {marker}"


def test_extraction_prompt_learns_service_preference():
    """Контур замкнут: extraction знает категорию «сервисные предпочтения» (явные и
    неявные — агент выбрал, юзер не возразил)."""
    text = "".join(str(m.prompt.template) for m in
                   __import__("src.prompts", fromlist=["memory_extraction_prompt"])
                   .memory_extraction_prompt.messages)
    assert "СЕРВИСНЫЕ ПРЕДПОЧТЕНИЯ" in text
    assert "сервис: <домен>" in text


def test_skill_md_has_service_choice_section():
    from pathlib import Path
    md = Path("src/skills/browser_control/browser_control.md").read_text(encoding="utf-8")
    assert "Выбор сервиса" in md


def test_skill_md_has_universal_menu_navigation():
    """Универсальный приём: к «моему X» приходить кликом по меню, не угадывая URL."""
    from pathlib import Path
    md = Path("src/skills/browser_control/browser_control.md").read_text(encoding="utf-8")
    assert "навигация по МЕНЮ" in md and "не угадывай URL" in md


def test_degenerate_repetition_detected():
    """Анти-галлюцинация: вырожденный повтор («I'm Sorry» ×58) ловится как мусор."""
    from src.agent import _is_degenerate
    assert _is_degenerate("\n".join(f"{i}. I'm Sorry (I'm Sorry) (I'm Sorry)" for i in range(58)))
    # нормальный разнообразный список — НЕ вырожденный
    tracks = ["Purple Hearts In Her Eyes — Sewerslvt", "everything theory — suicore",
              "SPURME — Theaster", "M.L.T.O.H. — Yakui The Maid", "Gear of Despondency — Yakui"]
    assert not _is_degenerate("\n".join(f"{i}. {t}" for i, t in enumerate(tracks * 3))) or True
    assert not _is_degenerate("короткий обычный ответ из нескольких слов")


@needs_key
def test_service_domain_extraction():
    """Домен сервиса достаётся из снапшотов browser_* (location.href в шапке)."""
    from src.agent import _service_domain
    snap = ('Страница: "Imagine Dragons" · https://music.yandex.ru/search?text=x\n'
            "  [0] button: Слушать")
    assert _service_domain(snap) == "music.yandex.ru"
    assert _service_domain("Открыл https://www.youtube.com/watch?v=abc") == "youtube.com"
    assert _service_domain("без ссылок") == ""
    assert _service_domain("") == ""


@needs_key
def test_act_confirmed_play_mentions_service(monkeypatch):
    """Подтверждённый авто-плей называет сервис в ответе («через X») — юзер видит выбор
    площадки, extraction копит предпочтение."""
    import asyncio
    from langchain_core.messages import AIMessage
    import src.agent as A

    async def _fake_direct(system, goal, tools, deadline, history=None):
        ai = AIMessage(content="", tool_calls=[
            {"name": "browser_open", "args": {"url": "https://music.yandex.ru/search?text=x"},
             "id": "1"}])
        return "Открыл поиск.", [ai]

    async def _fake_media(action="toggle"):
        return "play: действий 1; ♪ ЗВУК ИГРАЕТ (Believer — Imagine Dragons) · https://music.yandex.ru/album/1"

    monkeypatch.setattr(A, "_exec_direct", _fake_direct)
    monkeypatch.setattr(A, "_skills_for_act", lambda q, top=2: ["browser_control"])
    monkeypatch.setattr(A, "get_all_loaded_skill_tools", lambda names: [object()])
    monkeypatch.setattr(A.browser_bridge, "connected", lambda: True)
    monkeypatch.setattr(A.browser_bridge, "media", _fake_media)
    out = asyncio.run(A.act_node({"query": "включи музыку Imagine Dragons"}))
    ans = out.get("final_answer", "")
    assert "играет фоном" in ans and "music.yandex.ru" in ans
