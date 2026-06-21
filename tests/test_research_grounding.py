"""Ядро цели «надёжный поисковик»: запросы про реальные внешние факты (адреса/сайты/цены/
где купить/как оформить/лучшие в городе) заземляются веб-поиском, синтез строго из находок,
выдуманные URL вырезаются, анализ — headless (без кражи фокуса). Офлайн (без сети/LLM)."""
import src.graph.agent as A


def test_grounding_floor_catches_factual_queries():
    """Запросы-примеры юзера → требуют веб-заземления (нельзя из памяти — выдумает)."""
    for q in ["хочу дешевые брюки, где лучше взять?",
              "Самые лучшие суши в Москве - адреса, и сайты",
              "Как получить пособие по инвалидности?",
              "где купить дешёвый ноутбук",
              "лучшие кофейни Питера",
              "сколько стоит загранпаспорт"]:
        assert A._needs_web_grounding(q), q


def test_grounding_floor_catches_choice_and_compare():
    """Выбор товара и сравнение моделей/брендов → нужны текущие цены/специфы (не из памяти)."""
    for q in ["какой ноутбук купить до 50000", "какой выбрать пылесос",
              "сравни iPhone 16 и Samsung S24", "что подарить маме на день рождения"]:
        assert A._needs_web_grounding(q), q


def test_grounding_floor_skips_pure_reasoning():
    """Чистое рассуждение/болтовня/сравнение ПОНЯТИЙ — НЕ дёргаем поиск (модель знает сама)."""
    for q in ["реши уравнение 2x+3=7", "привет, как дела", "напиши хокку про осень",
              "переведи 'hello' на французский", "сравни рекурсию и итерацию",
              "чем отличается список от кортежа"]:
        assert not A._needs_web_grounding(q), q


def test_search_query_reformulation_chain_exists():
    """Сырой разговорный запрос → фокусный поисковый (магазины, не форумы) — есть цепочка."""
    import src.graph.agent as AA
    assert AA.search_query_chain is not None
    from src.llm.prompts import search_query_prompt
    text = "".join(str(m.prompt.template) for m in search_query_prompt.messages)
    assert "интернет-магазин" in text and "госуслуги" in text  # типы источников зашиты


def test_research_is_headless_no_focus_steal():
    """Факт-запрос БЕЗ физ-интента → руки ТОЛЬКО headless web_search (физ-вкладку, крадущую
    фокус, не открываем во время анализа — живой фидбек «отвлёкся на открытую ссылку»)."""
    picked = A._skills_for_act("Самые лучшие суши в Москве - адреса, и сайты")
    assert "web_search" in picked
    assert not (A._PHYSICAL_SKILLS & set(picked))


def test_play_keeps_physical_hands():
    """Воспроизведение — физ-интент: руки браузера остаются (музыка/видео не headless)."""
    assert A._wants_physical_browser("включи трек sewerslvt blooming iridescent flower")
    assert "browser_control" in A._skills_for_act("включи музыку Radiohead")


def test_strip_ungrounded_urls_removes_fabricated():
    """АНТИ-ВЫДУМКА URL: домен, которого НЕ было в реальной выдаче, вырезается; markdown-ссылка
    сводится к тексту. Заземлённый домен — остаётся как есть (живой баг: модель сочинила
    sakura-msk.ru/tanuki.ru по памяти)."""
    grounded = {"zoon.ru", "kp.ru"}
    ans = ("Лучшие: [Sakura](https://sakura-msk.ru) и [рейтинг КП](https://www.kp.ru/sushi). "
           "Ещё https://tanuki.ru и подборка https://zoon.ru/msk/sushi.")
    out = A._strip_ungrounded_urls(ans, grounded)
    assert "sakura-msk.ru" not in out      # выдуманный markdown-URL убран
    assert "tanuki.ru" not in out          # выдуманный голый URL убран
    assert "Sakura" in out                 # текст ссылки сохранён
    assert "kp.ru/sushi" in out            # реальный домен остался
    assert "zoon.ru/msk/sushi" in out      # реальный домен остался


def test_domains_of_extracts_hosts():
    doms = A._domains_of("см https://www.gosuslugi.ru/help и http://zoon.ru/x")
    assert "gosuslugi.ru" in doms and "zoon.ru" in doms
