"""/compact + статус-бар контекста + COMPACT.md (кумулятивный). Offline (без LLM)."""
import pytest

import src.interface.compact as cp


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def test_band_labels():
    assert cp.band_label(50_000) == "50k"
    assert cp.band_label(128_000) == "128k"
    assert cp.band_label(130_000) == "128k+"
    assert cp.band_label(256_000) == "256k"
    assert cp.band_label(300_000) == "256k+"
    assert cp.band_label(600_000) == "512k+"
    assert cp.band_label(1_000_000) == "1m"
    assert cp.band_label(2_000_000) == "1m"


def test_estimate_and_history_tokens():
    assert cp.estimate_tokens("x" * 400) == 100
    hist = [{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b" * 400}]
    assert cp.history_tokens(hist) == 200


def test_context_status_has_label_and_hint():
    s = cp.context_status(300_000)
    assert "256k+" in s and "/ 1M" in s and "/compact" in s


def test_should_auto_compact_at_1m():
    assert cp.should_auto_compact(999_999) is False
    assert cp.should_auto_compact(1_000_000) is True


def test_compact_md_cumulative_and_references(root):
    assert cp.read_compact() == ""
    n1 = cp.append_compact("первый срез: решили X")
    assert n1 == 1
    txt = cp.read_compact()
    assert "# COMPACT.md" in txt and "## Сжатие 1" in txt and "решили X" in txt
    n2 = cp.append_compact("второй срез: сделали Y")
    assert n2 == 2
    txt2 = cp.read_compact()
    assert "## Сжатие 2" in txt2 and "продолжает сжатия 1–1" in txt2  # ссылка на прошлое
    assert "решили X" in txt2 and "сделали Y" in txt2                  # кумулятивно


def test_gather_meta_reads_conventions(root):
    (root / "SEA.md").write_text("проект на langgraph, агент", encoding="utf-8")
    (root / "MCP.md").write_text("```yaml\nservers:\n  - name: fetch\n    command: uvx\n```\n",
                                 encoding="utf-8")
    meta = cp.gather_meta()
    assert "langgraph" in meta["project"]
    assert "fetch" in meta["user_mcp"]
