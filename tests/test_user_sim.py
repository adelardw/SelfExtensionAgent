"""Симуляция пользователей (user_sim): реестр персон, промпт, транскрипт, цикл диалога
(инжектируемые корутины — без LLM), агрегация. Offline."""
import asyncio

import pytest

from src.eval.user_sim import (PERSONAS, PersonaTurn, aggregate, format_transcript,
                               persona_system_prompt, run_dialogue)


def test_personas_registry_wellformed():
    ids = [p["id"] for p in PERSONAS]
    assert len(ids) == len(set(ids)) and len(ids) >= 3          # уникальны, персон достаточно
    for p in PERSONAS:
        for key in ("name", "profile", "style", "patience", "scenario"):
            assert p.get(key), f"{p['id']}: нет {key}"
        for key in ("goal", "opening", "success"):
            assert p["scenario"].get(key), f"{p['id']}: сценарий без {key}"
        assert 1 <= p["patience"] <= 6                          # потолок ходов разумный (бюджет)


def test_persona_prompt_carries_identity():
    p = PERSONAS[0]
    sp = persona_system_prompt(p)
    assert p["name"] in sp and p["scenario"]["goal"] in sp and p["style"] in sp
    assert "РОЛЬ пользователя" in sp                            # именно юзер, не ассистент


def test_format_transcript_roles_and_truncation():
    t = format_transcript([{"role": "user", "content": "привет"},
                           {"role": "assistant", "content": "х" * 3000}])
    assert "ПОЛЬЗОВАТЕЛЬ: привет" in t and "АССИСТЕНТ:" in t
    assert len(t) < 2000                                        # длинный ответ обрезан


def _turn(sat, done, msg="ещё вопрос", fb="норм"):
    return PersonaTurn(satisfaction=sat, feedback=fb, done=done, message=msg)


def test_run_dialogue_stops_on_done():
    calls = {"agent": 0, "persona": 0}

    async def agent_call(q, hist, tid):
        calls["agent"] += 1
        return f"ответ на: {q}"

    async def persona_step(p, transcript):
        calls["persona"] += 1
        return _turn(3, done=False) if calls["persona"] == 1 else _turn(5, done=True, msg="")

    res = asyncio.run(run_dialogue(agent_call, persona_step, PERSONAS[0], "t1"))
    assert calls["agent"] == 2 and res["turns"] == 2
    assert res["satisfaction"] == [3, 5]
    assert res["history"][0]["content"] == PERSONAS[0]["scenario"]["opening"]
    assert len(res["history"]) == 4                             # 2 пары user/assistant


def test_run_dialogue_respects_patience_cap():
    async def agent_call(q, hist, tid):
        return "ответ"

    async def persona_step(p, transcript):
        return _turn(2, done=False, msg="давай ещё")            # персона никогда не done

    res = asyncio.run(run_dialogue(agent_call, persona_step, PERSONAS[0], "t2", max_turns=2))
    assert res["turns"] == 2                                    # потолок, не бесконечный цикл


def test_run_dialogue_empty_message_ends():
    async def agent_call(q, hist, tid):
        return "ответ"

    async def persona_step(p, transcript):
        return _turn(4, done=False, msg="   ")                  # пустая реплика = завершение

    res = asyncio.run(run_dialogue(agent_call, persona_step, PERSONAS[0], "t3"))
    assert res["turns"] == 1


def test_aggregate_math():
    results = [
        {"satisfaction": [3, 5], "verdict": {"goal_achieved": True, "grounded": True,
                                             "depth": 4, "ux_issues": ["долго"]}},
        {"satisfaction": [2], "verdict": {"goal_achieved": False, "grounded": True,
                                          "depth": 2, "ux_issues": []}},
    ]
    agg = aggregate(results)
    assert agg["n"] == 2 and agg["achieved"] == 1 and agg["grounded"] == 2
    assert agg["avg_satisfaction"] == pytest.approx(10 / 3, abs=0.01)
    assert agg["avg_depth"] == 3.0 and agg["issues"] == ["долго"]


def test_aggregate_empty_safe():
    agg = aggregate([])
    assert agg["n"] == 0 and agg["avg_satisfaction"] == 0.0
