"""SE-Agent-lite ревизия/рекомбинация траекторий на полном ретрае."""
from src.graph.agent import _revision_hint


def test_first_attempt_no_hint():
    assert _revision_hint({}) == ""
    assert _revision_hint({"failed_trajectories": []}) == ""


def test_revision_demands_orthogonal_and_lists_failures():
    st = {"failed_trajectories": [
        {"approach": "веб-поиск ВВП на сайте X", "why_failed": "источник не открылся"},
        {"approach": "API Росстата", "why_failed": "схема ответа изменилась"}]}
    h = _revision_hint(st)
    assert "ортогональн" in h.lower()
    assert "веб-поиск ВВП на сайте X" in h and "API Росстата" in h
    assert "источник не открылся" in h


def test_recombination_carries_verified_prior():
    st = {"failed_trajectories": [{"approach": "a", "why_failed": "b"}],
          "prior_findings": "- Инфляция 2024: 8.5% (источник ЦБ)"}
    h = _revision_hint(st)
    assert "ПЕРЕИСПОЛЬЗУЙ" in h
    assert "Инфляция 2024: 8.5%" in h


def test_caps_to_last_three_failures():
    st = {"failed_trajectories": [{"approach": f"a{i}", "why_failed": "x"} for i in range(5)]}
    h = _revision_hint(st)
    assert "a4" in h and "a3" in h and "a2" in h  # последние 3
    assert "a0" not in h and "a1" not in h         # старые опущены
