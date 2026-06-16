"""Семантические подтверждения и уровни доверия HITL, CLI-конфиг, чистка тул-маркапа.
Офлайн: LLM-разбор мокается → проверяется фолбэк-маппинг и вся механика доверия."""
import asyncio

import pytest
from pydantic import BaseModel

from src import hitl, interaction, semantics
from src.semantics import parse_assent


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    interaction.reset_ledger()

    async def _no_llm(text, action):
        return None  # офлайн: всегда фолбэк-лексикон
    monkeypatch.setattr(semantics, "_llm_parse", _no_llm)
    monkeypatch.setattr(hitl, "_user_grants", {})       # чистые сессионные гранты (per-user)
    monkeypatch.setattr(hitl, "_config_grants", set())
    monkeypatch.setattr(hitl, "_work_mode", {})
    hitl.set_work_mode("manual")
    # гранты в тестах не персистим в реальный config.local.yml
    monkeypatch.setattr(hitl, "grant",
                        lambda key, persist=True, user_id=None: hitl._user_grants.setdefault("", set()).add(key))
    yield
    hitl.set_confirmer(None)


# ── parse_assent (офлайн-фолбэк) ────────────────────────────────────────────
def test_parse_plain_yes_no():
    assert parse_assent("да") == (True, "")
    assert parse_assent("Давай!") == (True, "")
    assert parse_assent("открывай") == (True, "")
    assert parse_assent("нет") == (False, "")
    assert parse_assent("не надо") == (False, "")


def test_parse_conditional_reason_redirect():
    d, note = parse_assent("да, но только не удаляй ничего")
    assert d is True and "не удаляй" in note
    d, note = parse_assent("нет, это опасно")
    assert d is False and note == "это опасно"
    d, note = parse_assent("лучше открой gmail")
    assert d is None and note == "лучше открой gmail"
    d, _ = parse_assent("да нет, не надо")  # «да» как частица
    assert d is False


def test_parse_reply_fallback_mapping():
    run = lambda t: asyncio.run(semantics.parse_reply(t))
    assert run("да") == ("yes", "")
    assert run("да, всегда") == ("always", "всегда")
    assert run("да, но в фоне")[0] == "condition"
    assert run("нет, личное") == ("no", "личное")
    assert run("ты зацикливаешься") == ("redirect", "ты зацикливаешься")
    assert run("")[0] == "no"  # пустой ответ — не разрешение


# ── HITL: уровни доверия ────────────────────────────────────────────────────
class _Args(BaseModel):
    url: str = ""


def _tool(name="open_url"):
    class _T:
        description = "test tool"
        args_schema = _Args

        async def ainvoke(self, kwargs):
            return f"OK {kwargs}"
    _T.name = name
    return _T()


def _wrapped(name="open_url"):
    return hitl.wrap_with_confirmation(_tool(name), "device_control")


def _ask_log():
    calls = []
    def conf(d):
        calls.append(d)
        return "да"
    return calls, conf


def test_readonly_tools_never_ask():
    calls, conf = _ask_log()
    hitl.set_confirmer(conf)
    out = asyncio.run(hitl.wrap_with_confirmation(_tool("analyze_screen"), "device_control")
                      .coroutine(url="x"))
    assert out.startswith("OK") and calls == []  # смотреть на экран — без вопроса


def test_always_grants_for_session():
    answers = iter(["да, всегда", "НЕ ДОЛЖНО СПРОСИТЬ"])
    asked = []
    def conf(d):
        asked.append(d)
        return next(answers)
    hitl.set_confirmer(conf)
    w = _wrapped()
    assert asyncio.run(w.coroutine(url="a")).startswith("OK")
    assert asyncio.run(w.coroutine(url="b")).startswith("OK")  # грант — второй раз молча
    assert len(asked) == 1 and hitl.is_granted("device_control.open_url")


def test_auto_mode_skips_all():
    calls, conf = _ask_log()
    hitl.set_confirmer(conf)
    hitl.set_auto(True)
    assert asyncio.run(_wrapped().coroutine(url="x")).startswith("OK")
    assert calls == []


def test_conditional_yes_asks_to_adjust():
    hitl.set_confirmer(lambda d: "да, но открой в фоновом окне")
    out = asyncio.run(_wrapped().coroutine(url="x"))
    assert "при условии" in out and "фоновом окне" in out
    assert hitl.REFUSAL_MARK not in out  # не жёсткий отказ — можно скорректировать
    ev = interaction.events()[0]
    assert ev["approved"] is False and "фоновом" in ev["note"]


def test_redirect_followed_and_no_with_reason():
    hitl.set_confirmer(lambda d: "лучше открой gmail")
    out = asyncio.run(_wrapped().coroutine(url="m.ya.ru"))
    assert "указание" in out and "gmail" in out and hitl.REFUSAL_MARK not in out
    hitl.set_confirmer(lambda d: "нет, это личное")
    out = asyncio.run(_wrapped().coroutine(url="x"))
    assert hitl.REFUSAL_MARK in out and "это личное" in out


def test_plain_yes_and_bool_compat():
    hitl.set_confirmer(lambda d: "ага")
    assert asyncio.run(_wrapped().coroutine(url="x")).startswith("OK")
    hitl.set_confirmer(lambda d: True)  # кнопки бота возвращают bool — совместимость
    assert asyncio.run(_wrapped().coroutine(url="x")).startswith("OK")


# ── CLI-конфиг: local поверх базового, запись только в local ────────────────
def test_cli_config_merge_and_persist(monkeypatch, tmp_path):
    from src import cli_config as cc
    base = tmp_path / "config.yml"
    base.write_text("agent:\n  habit_k: 3\ncli:\n  provider: openrouter\n", encoding="utf-8")
    monkeypatch.setattr(cc, "BASE", base)
    monkeypatch.setattr(cc, "LOCAL", tmp_path / "config.local.yml")
    assert cc.get_cli("provider") == "openrouter"
    cc.set_cli("provider", "ollama"); cc.set_cli("model", "qwen3:4b")
    assert cc.get_cli("provider") == "ollama" and cc.get_cli("model") == "qwen3:4b"
    # базовый config.yml НЕ тронут (комментарии/правда не перезаписываются)
    assert "ollama" not in base.read_text(encoding="utf-8")
    merged = cc.load_merged()
    assert merged.agent.habit_k == 3 and merged.cli.model == "qwen3:4b"


# ── чистка тул-маркапа (живой прогон: DSML утёк в ответ) ────────────────────
def test_strip_tool_markup_dsml_leak():
    from src.utils import strip_tool_markup
    leak = ('<｜DSML｜tool_calls> <｜DSML｜invoke name="open_url"> <｜DSML｜parameter name="url" '
            'string="true">https://music.yandex.ru/search?text=yakui+the+maid</｜DSML｜parameter> '
            '</｜DSML｜invoke> </｜DSML｜tool_calls>')
    assert strip_tool_markup(leak) == ""  # это не ответ — мусор вызова
    assert strip_tool_markup("Открыл Яндекс Музыку, включаю Yakui the Maid.") == \
        "Открыл Яндекс Музыку, включаю Yakui the Maid."
    mixed = f"Открываю поиск исполнителя. {leak}"
    out = strip_tool_markup(mixed)
    assert "DSML" not in out and "Открываю поиск" in out


def test_work_modes_three_states():
    """ручной / auto-accept (только подтверждения) / auto (полная автономия агента)."""
    assert hitl.set_work_mode("auto-accept") == "auto-accept"
    assert hitl.is_auto() and not hitl.full_auto()   # подтверждает сам, но спрашивает уточнения
    assert hitl.set_work_mode("auto") == "auto"
    assert hitl.is_auto() and hitl.full_auto()        # автономен целиком
    assert hitl.set_work_mode("manual") == "manual"
    assert not hitl.is_auto() and not hitl.full_auto()
    assert hitl.set_work_mode("мусор") == "manual"    # неизвестное → безопасный ручной
