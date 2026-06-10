"""ParamStore: overrides/few-shots/tool-descriptions — обратимо и с потолком."""
import pytest

from src.improve import prompt_store as ps


@pytest.fixture(autouse=True)
def tmp_params(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PARAMS_FILE", tmp_path / "params.json")


def test_override_roundtrip():
    assert ps.get_prompt("goal", "default") == "default"
    v1 = ps.save_override("goal", "новый промпт", rationale="test")
    assert v1 == 1
    assert ps.get_prompt("goal", "default") == "новый промпт"
    v2 = ps.save_override("goal", "ещё новее")
    assert v2 == 2
    assert ps.revert("goal") is True
    assert ps.get_prompt("goal", "default") == "default"
    assert ps.revert("goal") is False


def test_fewshots_dedupe_and_cap():
    for i in range(ps.MAX_FEWSHOTS + 4):
        ps.add_fewshot("step_execution", f"запрос номер {i} с уникальным хвостом", f"ответ {i}", score=i / 10)
    shots = ps.get_fewshots("step_execution", k=99)
    assert len(shots) <= ps.MAX_FEWSHOTS
    # топ по score: лучший пример сохранён
    assert shots[0]["score"] == max(s["score"] for s in shots)

    # дедуп по началу запроса: повтор не плодит копии
    ps.add_fewshot("r2", "один и тот же запрос", "a", 0.5)
    ps.add_fewshot("r2", "один и тот же запрос", "b", 0.9)
    shots = ps.get_fewshots("r2", k=99)
    assert len(shots) == 1
    assert shots[0]["answer"] == "b"


def test_format_fewshots_empty():
    assert "нет" in ps.format_fewshots("пусто").lower()


def test_tool_desc():
    assert ps.get_tool_desc("search_web", "дефолт") == "дефолт"
    ps.save_tool_desc("search_web", "обновлённое описание")
    assert ps.get_tool_desc("search_web", "дефолт") == "обновлённое описание"
