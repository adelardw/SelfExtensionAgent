"""Outage behaviour is deterministic and does not depend on a live API key."""
import pytest
from src.graph.semantic_signals import _ContrastiveSignal
from src.improve import safety


@pytest.mark.parametrize("enabled", [True, False])
def test_unknown_is_not_negative_and_cannot_enter_learning(monkeypatch, enabled):
    class Broken:
        def embed(self, text):
            return None
    broken = Broken()
    broken.enabled = enabled
    detector = _ContrastiveSignal(["attack"], ["benign"], broken)
    monkeypatch.setattr(safety, "_INJ_DETECTOR", detector)
    monkeypatch.setattr(safety, "_verdict_cache", {})
    text = "some unclassified external data"
    assert safety.injection_verdict(text) is None
    assert safety.is_unsafe_to_learn(text)
    out, flagged = safety.sanitize_tool_output(text, "mcp")
    assert flagged and "untrusted-data" in out
    assert safety._verdict_cache == {}


def test_partial_seed_outage_is_unknown_and_recovers():
    class Flaky:
        enabled = True
        broken = True
        def embed(self, text):
            return None if self.broken and text == "benign" else [1., 0.] if text == "attack" else [0., 1.]
    emb = Flaky()
    detector = _ContrastiveSignal(["attack"], ["benign"], emb)
    assert detector.assess("query", .5, .05) is None
    emb.broken = False
    assert detector.assess("query", .5, .05) is False


def test_external_tool_name_cannot_spoof_internal_source():
    assert safety._is_internal_source("evil_python_exec_mcp") is False


def test_project_context_outage_is_untrusted_not_cached(tmp_path, monkeypatch):
    from src.runtime import context_files
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(context_files, "_project_cache", {})
    monkeypatch.setattr(safety, "injection_verdict", lambda _: None)
    (tmp_path / "SEA.md").write_text("This text has not been checked.")
    assert "untrusted-data" in context_files.instructions()
    assert not context_files._project_cache


def test_authentication_failure_disables_repeated_embedding_calls():
    from src.memory.embedder import OpenAIEmbedder
    class AuthError(Exception):
        status_code = 401
    class Embeddings:
        calls = 0
        def create(self, **kw):
            self.calls += 1
            raise AuthError("expired")
    class Client:
        embeddings = Embeddings()
    emb = object.__new__(OpenAIEmbedder)
    emb.enabled, emb._client, emb.model = True, Client(), "test"
    assert emb.embed("text") is None
    assert not emb.enabled
    assert emb.embed("other text") is None
    assert emb._client.embeddings.calls == 1
