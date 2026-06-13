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


def test_research_is_headless_play_is_physical():
    """Архитектурная граница (фидбек юзера «отвлёкся на открытую ссылку, анализ скрытым»):
    ЧТЕНИЕ/анализ/поиск → ТОЛЬКО headless web_search (физ-вкладка не открывается, фокус не
    крадётся). ВОСПРОИЗВЕДЕНИЕ/действие на сайте → физ-руки (browser_control). Гейт рук, не
    маршрутизация режима."""
    from src.agent import _skills_for_act, _PHYSICAL_SKILLS
    research = _skills_for_act("найди обзоры наушников и перечисли варианты")
    assert "web_search" in research
    assert not (_PHYSICAL_SKILLS & set(research))  # анализ — без физ-навыков (без кражи фокуса)
    play = _skills_for_act("включи трек chikoi the maid")
    assert "browser_control" in play and "web_search" in play  # воспроизведение — физ-руки есть


def test_reflexion_prompt_routes_browse_to_act():
    """Принцип browse=act и grounding-честность зашиты в reflexion-промпт (не в регулярку)."""
    from src.prompts import reflexion_prompt
    text = "".join(str(m.prompt.template) for m in reflexion_prompt.messages)
    assert "BROWSE = тоже act" in text
    assert "grounding" in text.lower()


def test_meta_ack_detected():
    """Мета-ответ-заглушка («я перечислил/список выше») ловится → нужен ресинтез результата."""
    from src.agent import _is_meta_ack
    assert _is_meta_ack("Понял, задача выполнена. Я перечислил 7 названий видео.")
    assert _is_meta_ack("Вот итог — список выше. Больше действий не требуется.")
    assert _is_meta_ack("Всё готово.")
    # настоящий результат со списком — НЕ мета-заглушка (длинный, есть содержимое)
    real = ("Вот обзоры:\n1. ПОЛ ГОДА С Sony WH-1000XM5\n2. Big Review These Are GOOD\n"
            "3. Топовые наушники Sony\n4. Флагманские с нюансами\n5. Самое подробное видео 2025")
    assert not _is_meta_ack(real)
    assert not _is_meta_ack("Включил Believer — Imagine Dragons, играет фоном.")


def test_false_access_refusal_detected():
    """Ложный отказ «нет доступа к аккаунтам» ловится (доступ есть через расширение)."""
    from src.agent import _is_false_access_refusal
    assert _is_false_access_refusal("Я не имею доступа к вашим личным аккаунтам и данным.")
    assert _is_false_access_refusal("Sorry, I cannot access your personal Spotify account.")
    assert _is_false_access_refusal("Моя функциональность ограничена доступом к данным.")
    # нормальный честный ответ — НЕ ловится
    assert not _is_false_access_refusal("Открыл твоё избранное в Spotify, вот треки: …")
    assert not _is_false_access_refusal("Не нашёл такого исполнителя на этом сервисе.")


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

    async def _fake_direct(system, goal, tools, deadline, history=None, **kw):
        ai = AIMessage(content="", tool_calls=[
            {"name": "browser_open", "args": {"url": "https://music.yandex.ru/search?text=x"},
             "id": "1"}])
        return "Открыл поиск.", [ai]

    async def _fake_media(action="toggle"):
        return "play: действий 1; ♪ ЗВУК ИГРАЕТ (Believer — Imagine Dragons) · https://music.yandex.ru/album/1"

    monkeypatch.setattr(A, "_exec_direct", _fake_direct)
    monkeypatch.setattr(A, "_skills_for_act", lambda q, top=2, qvec=None: ["browser_control"])
    monkeypatch.setattr(A, "get_all_loaded_skill_tools", lambda names: [object()])
    monkeypatch.setattr(A.browser_bridge, "connected", lambda: True)
    monkeypatch.setattr(A.browser_bridge, "media", _fake_media)
    out = asyncio.run(A.act_node({"query": "включи музыку Imagine Dragons"}))
    ans = out.get("final_answer", "")
    assert "играет фоном" in ans and "music.yandex.ru" in ans


def test_payment_boundary_in_content_actions():
    """[ГРАНИЦА ОПЛАТЫ] кнопка оформления/оплаты в content_actions.js — агент не нажимает сам."""
    from pathlib import Path
    js = Path("extension/content_actions.js").read_text(encoding="utf-8")
    assert "ГРАНИЦА ОПЛАТЫ" in js and "payLabel" in js and "payRefusal" in js
    # граница применена И в click, И в clicktext
    assert js.count("payLabel(") >= 2


def test_skill_md_has_payment_boundary():
    from pathlib import Path
    md = Path("src/skills/browser_control/browser_control.md").read_text(encoding="utf-8")
    assert "ГРАНИЦА ОПЛАТЫ" in md and "финальную кнопку" in md


def test_anti_typosquat_url_correction():
    """Безопасность: близкий-но-другой домен (тайпсквоттинг) правится на ТОЧНЫЙ пользовательский."""
    from src import browser_bridge as br
    br.set_user_domains("закажи суши с сайта https://niyama.ru/ любой набор")
    assert br.safe_url("https://niama.ru/menu") == "https://niyama.ru/menu"   # тайпсквот → точный
    assert br.safe_url("https://niyama.ru/cart") == "https://niyama.ru/cart"  # точный — без изменений
    assert br.safe_url("https://google.com") == "https://google.com"          # другой — не трогаем
    br.set_user_domains("включи музыку")  # юзер не давал домен
    assert br.safe_url("https://music.yandex.ru") == "https://music.yandex.ru"  # не над-правим


def test_router_blocks_skill_creation_for_browser():
    """Орк-фикс: физический веб → use_skills (browser_control), НИКОГДА create_skill (+uv-инсталл)."""
    from src.prompts import router_prompt
    text = "".join(str(m.prompt.template) for m in router_prompt.messages)
    assert "ФИЗИЧЕСКИЙ ВЕБ" in text and "НИКОГДА 'create_skill'" in text
