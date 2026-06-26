"""Раздел M — self_model: env-id, overlay (стена Type-2), личность, session-commit/diff, self-model.

Всё ДЕТЕРМИНИРОВАННО, БЕЗ API — юниты на несущую логику M0 (тесты теперь пишем, в отличие от
голого composer). data/ пишется в tmp (chdir), реестр навыков фиксируем (иначе читал бы src/skills).
"""
import pytest

from src.runtime import self_model as SM


@pytest.fixture(autouse=True)
def _tmp_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(SM, "_safe_registry",
                        lambda: {"alpha": {"description": "a"}, "beta": {"description": "b"}})
    yield


# ── env-id: изоляция по умолчанию, склейка только по устойчивому якорю ──
def test_env_id_isolation_without_anchor():
    a = SM.resolve_env_id("web", "", run_id="r1")
    b = SM.resolve_env_id("web", "", run_id="r2")
    assert a != b and a.startswith("ephemeral:")          # без якоря НЕ склеиваем разные сеансы


def test_env_id_continuity_with_anchor():
    a = SM.resolve_env_id("chat", "thread-7", run_id="r1")
    b = SM.resolve_env_id("chat", "thread-7", run_id="r2")
    assert a == b == "chat:thread-7"                       # тот же якорь = тот же проект


def test_env_id_code_hashes_path():
    e = SM.resolve_env_id("code", "/secret/repo/path")
    assert e.startswith("code:") and "/secret/" not in e   # путь репо не светится


# ── overlay (Type 1) со стеной к Type 2 ──
def test_overlay_roundtrip_and_revert():
    env = "chat:t1"
    assert SM.get_overlay(env) == ""
    SM.save_overlay(env, "коротко и со ссылками", "из фидбека")
    assert "ссылками" in SM.get_overlay(env)
    assert SM.revert_overlay(env) is True
    assert SM.get_overlay(env) == ""                       # самоправка всегда обратима


def test_type2_wall_blocks_frozen_role():
    for role in ("reflexion", "synthesize", "verify"):
        with pytest.raises(ValueError):
            SM.save_overlay(role, "взлом базы")            # Type-2 физически не оптимизируется
    assert SM.is_type1("env:chat:t1") is True
    assert SM.is_type1("reflexion") is False


# ── личность: абстрактный дистиллят, сырьё/PII отсекается ──
def test_personality_accepts_abstract_rejects_raw():
    assert SM.add_personality_note("ценит развёрнутые ответы со ссылками") is True
    assert SM.add_personality_note("почта владельца ivan@example.com") is False   # PII
    assert SM.add_personality_note("счёт 1234567890") is False                    # сырое число
    out = SM.format_personality()
    assert "развёрнутые" in out and "example.com" not in out


def test_personality_dedup_and_cap():
    for i in range(50):
        SM.add_personality_note(f"черта про стиль вариант {chr(97 + i % 26)}-{i // 26}")
    assert len(SM._load_personality()["notes"]) <= SM.MAX_PERSONALITY


# ── session-commit + инкрементальный diff ──
def test_session_commit_and_diff(monkeypatch):
    env = "code:abc"
    d0 = SM.diff_since_last(env)
    assert d0["first_session"] is True and set(d0["added"]) == {"alpha", "beta"}
    SM.session_commit(env, primitives=["recall", "reason"])
    monkeypatch.setattr(SM, "_safe_registry",
                        lambda: {"alpha": {}, "beta": {}, "gamma": {}})        # появился навык
    d1 = SM.diff_since_last(env)
    assert d1["first_session"] is False and d1["added"] == ["gamma"] and d1["removed"] == []


# ── self-model: 5 граней, детерминированно, без сети ──
class _FakeStore:
    def recall_scored(self, uid, q, qvec=None):
        return ("факт: X=42", 0.5)

    def format_profile(self, uid):
        return "[ВНУТРЕННЕЕ]\n- role: аналитик"


def test_build_self_model_facets():
    SM.save_overlay("chat:t9", "будь краток")
    sm = SM.build_self_model(env_id="chat:t9", store=_FakeStore(), user_id="u1",
                             query="вопрос", qvec=[0.1, 0.2])
    assert "Моя структура" in sm            # (1) структура + цена примитивов
    assert "alpha" in sm and "beta" in sm    # (2) способности
    assert "X=42" in sm                      # (3) знания (recall, без сети)
    assert "аналитик" in sm                  # (4) собеседник
    assert "будь краток" in sm               # overlay среды (Type 1)
