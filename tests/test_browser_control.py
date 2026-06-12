"""browser_control: структурное управление браузером — загрузка, доверие, подбор в act.
Офлайн (без запуска Chromium: live-состояние в src.browser_session импортируется лениво)."""
import os

import pytest

needs_key = pytest.mark.skipif(
    not (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="нужен API-ключ: llm строится на импорте src.agent",
)

EXPECTED = {"browser_open", "browser_see", "browser_click", "browser_type",
            "browser_press", "browser_scroll"}


def test_skill_module_loads_without_browser():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bc", "src/skills/browser_control/browser_control.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # импорт НЕ поднимает Chromium (ленивые импорты)
    names = {getattr(getattr(m, n), "name", "") for n in dir(m)}
    assert EXPECTED <= names


def test_browser_see_is_readonly_no_confirm():
    from src import hitl
    assert "browser_see" in hitl.READONLY_TOOLS or "browser_see" in hitl._DEFAULT_READONLY
    # сами действия — под подтверждением (skills.confirm)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config.yml")
    assert "browser_control" in list(cfg.skills.confirm)


@needs_key
def test_act_picks_browser_hands():
    from src.agent import _skills_for_act
    picked = _skills_for_act("включи трек chikoi the maid")
    assert "browser_control" in picked


def test_browser_media_tool_exists_and_trusted():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bc2", "src/skills/browser_control/browser_control.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert getattr(m.browser_media, "name", "") == "browser_media"
    from src import hitl
    # «поставь на паузу» по просьбе юзера не должно переспрашиваться
    assert "browser_media" in hitl._DEFAULT_READONLY


def test_full_media_toolset_and_trust():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bc3", "src/skills/browser_control/browser_control.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    names = {getattr(getattr(m, n), "name", "") for n in dir(m)}
    assert {"browser_read", "browser_media"} <= names
    from src import hitl
    # смотреть/читать/управлять воспроизведением по просьбе — без подтверждений
    assert {"browser_see", "browser_read", "browser_media"} <= hitl._DEFAULT_READONLY


def test_default_backend_is_system_chrome():
    """Зафиксировано: по умолчанию системный Chrome пользователя, не Chromium."""
    import src.browser_session as bs
    # дефолт читается из конфига с фолбэком 'system' — проверяем фолбэк-значение в коде
    import inspect
    src = inspect.getsource(bs._ensure_page)
    assert '"system"' in src and 'channel="chrome"' in src  # дефолт + реальный Chrome


def test_background_focus_helpers_safe_offmac(monkeypatch):
    """Фоновый режим: захват/возврат фокуса не падают и no-op вне macOS."""
    import src.browser_session as bs
    monkeypatch.setattr(bs.platform, "system", lambda: "Linux")
    bs._capture_front()      # no-op, без исключений
    bs._restore_front()
    assert bs._background() in (True, False)  # дефолт читается


def test_background_default_on():
    """По умолчанию фоновый режим ВКЛ (окно не лезет на экран)."""
    import src.browser_session as bs
    import inspect
    assert "browser_background" in inspect.getsource(bs._background)
    # _call оборачивает возврат фокуса
    assert "_restore_front" in inspect.getsource(bs._call)


def test_routing_prefers_extension_when_connected(monkeypatch):
    """browser_* идут в расширение (твой браузер), если оно подключено; иначе — playwright."""
    import asyncio
    import src.browser_bridge as br
    import src.browser_session as bs
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bc4", "src/skills/browser_control/browser_control.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    calls = {"ext": 0}
    monkeypatch.setattr(br, "connected", lambda: True)
    monkeypatch.setattr(br, "ensure_server", lambda: None)
    async def _ext_open(url): calls["ext"] += 1; return f"ext:{url}"
    monkeypatch.setattr(br, "open_url", _ext_open)
    out = asyncio.run(m.browser_open.ainvoke({"url": "music.yandex.ru"}))
    assert out.startswith("ext:") and calls["ext"] == 1  # подключено → твой браузер


def test_bridge_token_and_server_boot():
    from src import browser_bridge as br
    br.ensure_server()
    assert len(br.token()) >= 16
    assert br._thread is not None and br._thread.is_alive()


def test_no_extension_returns_install_hint_not_sandbox(monkeypatch):
    """Физический веб без расширения → просьба поставить, НЕ песочное окно/подпроцесс."""
    import asyncio
    import src.browser_bridge as br
    import src.browser_session as bs
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bc5", "src/skills/browser_control/browser_control.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    monkeypatch.setattr(br, "connected", lambda: False)
    monkeypatch.setattr(br, "ensure_server", lambda: None)
    # session НЕ должен вызываться при дефолтном backend=extension
    def _boom(*a, **k): raise AssertionError("песочное окно не должно открываться")
    monkeypatch.setattr(bs, "open_url", _boom)
    from src import cli_config
    monkeypatch.setattr(cli_config, "get_cli", lambda k, d=None: None)  # backend=extension (деф)
    out = asyncio.run(m.browser_open.ainvoke({"url": "music.yandex.ru"}))
    assert "НУЖНО РАСШИРЕНИЕ" in out and "extension/" in out


def test_window_backend_opt_in_uses_session(monkeypatch):
    """Power-user явно выбрал окно (cli.browser_backend='window') → playwright-сессия."""
    import asyncio
    import src.browser_bridge as br
    import src.browser_session as bs
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bc6", "src/skills/browser_control/browser_control.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(br, "connected", lambda: False)
    monkeypatch.setattr(br, "ensure_server", lambda: None)
    monkeypatch.setattr(bs, "open_url", lambda url: f"window:{url}")
    from src import cli_config
    monkeypatch.setattr(cli_config, "get_cli", lambda k, d=None: "window" if k == "browser_backend" else None)
    out = asyncio.run(m.browser_open.ainvoke({"url": "music.yandex.ru"}))
    assert out == "window:https://music.yandex.ru" or out.startswith("window:")


def test_closed_browser_launches_and_waits(monkeypatch):
    """Закрытый браузер + просьба → агент поднимает браузер, ждёт расширение, потом действие."""
    import asyncio
    import src.browser_bridge as br
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bc7", "src/skills/browser_control/browser_control.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    state = {"connected": False, "launched": 0}
    monkeypatch.setattr(br, "ensure_server", lambda: None)
    monkeypatch.setattr(br, "connected", lambda: state["connected"])
    def _launch():
        state["launched"] += 1; state["connected"] = True; return True
    monkeypatch.setattr(br, "launch_browser", _launch)
    monkeypatch.setattr(br, "wait_connected", lambda t=12.0: state["connected"])
    async def _open(url): return f"ext:{url}"
    monkeypatch.setattr(br, "open_url", _open)
    from src import cli_config
    monkeypatch.setattr(cli_config, "get_cli", lambda k, d=None: None)
    out = asyncio.run(m.browser_open.ainvoke({"url": "music.yandex.ru"}))
    assert state["launched"] == 1 and out.startswith("ext:")  # подняли браузер → действие


def test_bridge_chat_handler_roundtrip():
    """Чат из расширения прогоняется через зарегистрированный обработчик графа."""
    import asyncio
    from src import browser_bridge as br

    async def _handler(text): return f"ответ на: {text}"
    br.set_chat_handler(_handler)
    # _serve_chat шлёт результат в ws — проверяем через прямой вызов обработчика
    assert asyncio.run(br._chat_handler("привет")) == "ответ на: привет"
    br.set_chat_handler(None)


def test_act_does_not_lie_about_playback(monkeypatch):
    """act не заявляет «играет», если ни один тул не подтвердил воспроизведение."""
    import asyncio
    from langchain_core.messages import AIMessage, ToolMessage
    import src.agent as A

    async def _fake_direct(system, goal, tools, deadline, history=None):
        ai = AIMessage(content="", tool_calls=[{"name": "browser_click", "args": {"item": 3}, "id": "1"}])
        tm = ToolMessage(content="Кликнул [3]. Страница: 'Поиск'", tool_call_id="1")  # НЕ играет
        return "Включил трек, играет.", [ai, tm]
    monkeypatch.setattr(A, "_exec_direct", _fake_direct)
    monkeypatch.setattr(A, "_skills_for_act", lambda q, top=2: ["browser_control"])
    monkeypatch.setattr(A, "get_all_loaded_skill_tools", lambda names: [object()])
    out = asyncio.run(A.act_node({"query": "включи трек X"}))
    assert "НЕ пошл" in out["final_answer"] and "mode" not in out
    assert "Что сейчас на странице" in out["final_answer"]  # показываем реальный снапшот


def test_act_accepts_confirmed_playback(monkeypatch):
    """С подтверждением «ЗВУК ИГРАЕТ» в результате тула — успех принимается."""
    import asyncio
    from langchain_core.messages import AIMessage, ToolMessage
    import src.agent as A

    async def _fake_direct(system, goal, tools, deadline, history=None):
        ai = AIMessage(content="", tool_calls=[{"name": "browser_media", "args": {"action": "play"}, "id": "1"}])
        tm = ToolMessage(content="play: затронуто 1/1; ♪ ЗВУК ИГРАЕТ", tool_call_id="1")
        return "Включил трек, играет.", [ai, tm]
    monkeypatch.setattr(A, "_exec_direct", _fake_direct)
    monkeypatch.setattr(A, "_skills_for_act", lambda q, top=2: ["browser_control"])
    monkeypatch.setattr(A, "get_all_loaded_skill_tools", lambda names: [object()])
    out = asyncio.run(A.act_node({"query": "включи трек X"}))
    assert out.get("final_answer", "").startswith("Включил") and "mode" not in out
