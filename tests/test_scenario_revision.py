"""СЦЕНАРНАЯ валидация ревизии/рекомбинации: реально гоняем decompose_node и ловим, что в decompose
доходит ОРТОГОНАЛЬНАЯ инструкция + верифицированные прошлые результаты, а рецепт-шорткат на ретрае
пропускается. Мокнут только structured-LLM-вызов (его вход и захватываем)."""
import asyncio

import pytest

import src.graph.agent as A


def _drive(monkeypatch, state):
    captured = {}

    async def fake_structured(role, schema, sysvars, query):
        captured["role"] = role
        captured.update(sysvars)

        class _S:
            subtasks = []
            reasoning = "stub"
        return _S()

    monkeypatch.setattr(A, "_structured", fake_structured)
    base = {"query": "аналитика рынка РФ", "selected_skills": []}
    asyncio.run(A.decompose_node({**base, **state}))
    return captured


def test_first_attempt_has_no_revision(monkeypatch):
    cap = _drive(monkeypatch, {})
    assert cap, "decompose должен был вызвать structured-LLM"
    assert "ортогональн" not in cap.get("capability_hint", "").lower()


def test_retry_injects_orthogonal_and_prior(monkeypatch):
    cap = _drive(monkeypatch, {
        "failed_trajectories": [{"approach": "поиск на сайте Росстата", "why_failed": "страница не открылась"}],
        "prior_findings": "- Инфляция 2024: 8.5% (источник ЦБ)",
    })
    hint = cap.get("capability_hint", "")
    assert "ортогональн" in hint.lower()                       # РЕВИЗИЯ: иной подход
    assert "поиск на сайте Росстата" in hint                   # перечислен провал
    assert "Инфляция 2024: 8.5%" in hint and "ПЕРЕИСПОЛЬЗУЙ" in hint  # РЕКОМБИНАЦИЯ


def test_recipe_shortcut_skipped_on_retry(monkeypatch):
    """С recipe_id, НО при провале прошлой попытки — рецепт-шорткат пропускается, идём в полный
    decompose (рецепт уже провалился → нужен иной план), и туда доходит ревизия."""
    called = {"get_recipe": 0}
    monkeypatch.setattr(A.memory_store, "get_recipe",
                        lambda rid: called.update(get_recipe=called["get_recipe"] + 1) or None)
    cap = _drive(monkeypatch, {
        "recipe_id": 123,
        "failed_trajectories": [{"approach": "рецепт прошлого решения", "why_failed": "не сошлось"}],
    })
    assert called["get_recipe"] == 0          # шорткат к рецепту НЕ тронут
    assert cap, "ушли в полный decompose"
    assert "ортогональн" in cap.get("capability_hint", "").lower()
