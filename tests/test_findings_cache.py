"""Findings-кэш: чекпоинтер несёт session_findings между ходами (амортизация follow-up).

Фича reflect→recall держится на том, что кастомное поле state переживает ход через чекпоинтер
и изолировано по thread_id. Если апдейт LangGraph это сломает — кэш тихо перестанет работать
(follow-up снова уйдёт в тяжёлый ризонинг). Этот тест ловит регресс на мини-графе (без LLM).
"""
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class _S(TypedDict, total=False):
    query: str
    session_findings: str
    seen: str


def _build():
    def recall(s):  # как наш recall_node: ЧИТАЕТ findings, принесённый чекпоинтером
        return {"seen": s.get("session_findings", "")}

    def reflect(s):  # как наш reflect_node: тяжёлый турн ПИШЕТ findings, лёгкий — нет
        return {"session_findings": f"F::{s['query']}"} if s["query"].startswith("heavy") else {}

    g = StateGraph(_S)
    g.add_node("recall", recall)
    g.add_node("reflect", reflect)
    g.add_edge(START, "recall")
    g.add_edge("recall", "reflect")
    g.add_edge("reflect", END)
    return g.compile(checkpointer=MemorySaver())


def test_findings_survive_turn_and_isolated_by_thread():
    app = _build()
    t1 = {"configurable": {"thread_id": "t1"}}

    r1 = app.invoke({"query": "heavy: analyze repo"}, t1)
    assert r1["seen"] == ""                       # первый ход — находок ещё нет

    r2 = app.invoke({"query": "follow-up"}, t1)
    assert r2["seen"] == "F::heavy: analyze repo"  # follow-up ВИДИТ находки прошлого хода

    # лёгкий ход не затёр находки — они живут до следующего тяжёлого
    r3 = app.invoke({"query": "another follow-up"}, t1)
    assert r3["seen"] == "F::heavy: analyze repo"

    # другой тред — полная изоляция (находки не протекают между чатами)
    r_other = app.invoke({"query": "x"}, {"configurable": {"thread_id": "t2"}})
    assert r_other["seen"] == ""


def test_reflexion_prompt_has_findings_reuse_rule():
    """Промпт reflexion должен явно учить ронять режим при наличии находок (downgrade)."""
    from src.prompts import reflexion_prompt

    text = str(reflexion_prompt)
    assert "Уже проработано" in text and "ЛЁГКИЙ режим" in text


def test_relevant_findings_picks_semantically_closest():
    """Из коллекции находок recall впрыскивает СЕМАНТИЧЕСКИ близкие к запросу (top-k), не все."""
    from src.agent import _relevant_findings

    coll = [
        {"query": "repo", "summary": "REPO-FINDINGS", "emb": [1.0, 0.0, 0.0]},
        {"query": "weather", "summary": "WEATHER-FINDINGS", "emb": [0.0, 1.0, 0.0]},
    ]
    # запрос близок к repo → впрыскивается находка про repo, не про погоду
    out = _relevant_findings({"session_findings": coll, "query_emb": [1.0, 0.0, 0.0]}, k=1)
    assert out == "REPO-FINDINGS"
    # запрос близок к weather
    out2 = _relevant_findings({"session_findings": coll, "query_emb": [0.0, 1.0, 0.0]}, k=1)
    assert out2 == "WEATHER-FINDINGS"


def test_relevant_findings_below_gate_and_fallbacks():
    from src.agent import _relevant_findings

    coll = [{"query": "x", "summary": "X", "emb": [1.0, 0.0]}]
    # ортогональный запрос (sim=0 < gate) → ничего не впрыскиваем (не шумим)
    assert _relevant_findings({"session_findings": coll, "query_emb": [0.0, 1.0]}, k=2) == ""
    # нет эмбеддинга запроса → graceful: последняя находка
    assert _relevant_findings({"session_findings": coll, "query_emb": None}) == "X"
    # старый str-формат / пусто → пусто
    assert _relevant_findings({"session_findings": "old-string"}) == ""
    assert _relevant_findings({}) == ""
