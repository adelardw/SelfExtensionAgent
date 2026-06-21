"""Research-слой: схемы, извлечение URL, сборка тула, мок end-to-end. Офлайн."""
import asyncio

from langchain_core.runnables import RunnableLambda

from src.data.research import (FactCheck, ResearchPlan, _URL_RE, agentic_research,
                          make_deep_research_tool)


def test_url_extraction():
    txt = "1. Title\n   https://a.com/page\n   snippet\n2. T2\n   http://b.org/x)"
    urls = _URL_RE.findall(txt)
    assert "https://a.com/page" in urls and "http://b.org/x" in urls


def test_deep_research_tool_builds():
    t = make_deep_research_tool()
    assert t.name == "deep_research" and t.coroutine is not None


def test_agentic_research_chains_and_verifies(monkeypatch):
    # мок LLM: план из 2 под-вопросов, верификация found=true, синтез/реформуляция — эхо
    plan = ResearchPlan(subquestions=["кто выиграл", "столица победителя"])

    class _Fast:
        def with_structured_output(self, schema):
            if schema is ResearchPlan:
                return RunnableLambda(lambda _x: plan)
            return RunnableLambda(lambda _x: FactCheck(found=True, fact="Франция/Париж", confidence=0.95))

        async def _ainvoke(self, _x):
            class R:  # реформуляция и синтез
                content = "Франция → Париж"
            return R()

    fast = _Fast()
    # prompt | fast  → RunnableSequence; .ainvoke вызовет fast.ainvoke
    fast.ainvoke = fast._ainvoke
    monkeypatch.setattr("src.data.research.chat", lambda *a, **k: fast)

    # search/browse — подменяем сам объект тула фейком с .invoke (без сети)
    class _FakeTool:
        def __init__(self, ret):
            self._ret = ret

        def invoke(self, _x):
            return self._ret

    import src.skills.web_search.web_search as W
    monkeypatch.setattr(W, "search_web", _FakeTool("1. T\n   https://x.com/p\n   s"))
    # research читает через _page_text_urllib (не browse) — мокаем его, чтобы остаться офлайн
    monkeypatch.setattr(W, "_page_text_urllib", lambda _u: ("T", "Франция выиграла; столица Париж"))

    res = asyncio.run(agentic_research("multi-hop вопрос"))
    assert res["total"] == 2 and res["verified"] == 2
    assert all(f["found"] for f in res["facts"])
