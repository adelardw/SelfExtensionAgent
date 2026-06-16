"""`.sea/` workspace (L1): init создаёт каталог+конвенции, лог решений NO-OP без init
(аддитивно), после init пишет/читает решения. Offline."""
import pytest

import src.sea_workspace as sw


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def test_log_decision_noop_without_init(root):
    assert sw.is_initialized() is False
    sw.log_decision("skill.tool(x=1)", approved=False)  # не падает, ничего не пишет
    assert sw.decisions() == []


def test_init_creates_dir_and_conventions(root):
    created = sw.init()
    assert sw.is_initialized()
    assert (root / ".sea" / "history").is_dir()
    assert (root / ".sea" / "README.md").exists()
    for f in ("SEA.md", "MEMORY.md", "MCP.md"):
        assert (root / f).exists(), f
    assert ".sea" in " ".join(created) and "SEA.md" in created


def test_init_sea_md_has_repo_map(root):
    # фейковый репо: python-файлы + манифест → скан должен их увидеть
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x=1", encoding="utf-8")
    (root / "src" / "b.py").write_text("y=2", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0'\n[project.scripts]\nmycli='m:f'\n", encoding="utf-8")
    sw.init()
    sea = (root / "SEA.md").read_text(encoding="utf-8")
    assert "карта" in sea.lower() and "Python" in sea          # стек распознан
    assert "`src/`" in sea                                      # структура
    assert "mycli" in sea                                       # команда из pyproject.scripts


def test_scan_repo_robust_on_empty(root):
    out = sw.scan_repo(root)
    assert out.startswith("# SEA.md") and "Стек" in out         # не падает на пустом


def test_init_idempotent_no_overwrite(root):
    sw.init()
    (root / "SEA.md").write_text("МОИ правки", encoding="utf-8")  # юзер отредактировал
    created2 = sw.init()
    assert created2 == []  # ничего не создано повторно
    assert (root / "SEA.md").read_text(encoding="utf-8") == "МОИ правки"  # не перезаписан


def test_log_and_read_decisions_after_init(root):
    sw.init()
    sw.log_decision("file_edit(a.py)", approved=True, kind="yes")
    sw.log_decision("shell(rm)", approved=False, kind="no", note="опасно")
    d = sw.decisions()
    assert len(d) == 2 and d[0]["approved"] is True and d[1]["approved"] is False
    assert d[1]["note"] == "опасно"
