"""Root-convention файлы (SEA.md/MEMORY.md/SKILL.md/MCP.md) — дискавери из корня проекта.
КЛЮЧЕВОЕ: нет файлов → пусто (аддитивно, ноль изменения поведения). MCP.md → реестр серверов.
Offline, без сети/LLM."""
import pytest

import src.runtime.context_files as cf


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def test_no_files_is_noop(root):
    assert cf.instructions() == ""
    assert cf.mcp_servers() == []


def test_sea_and_skill_instructions(root):
    (root / "SEA.md").write_text("Всегда отвечай со ссылками на источники.", encoding="utf-8")
    (root / "SKILL.md").write_text("Навык web_search: глубокий поиск.", encoding="utf-8")
    instr = cf.instructions()
    assert "SEA.md" in instr and "ссылками на источники" in instr
    assert "SKILL.md" in instr and "web_search" in instr


def test_mcp_registry_yaml_block(root):
    (root / "MCP.md").write_text(
        "# Мои MCP-серверы\n\n"
        "```yaml\n"
        "servers:\n"
        "  - name: fetch\n"
        "    command: uvx\n"
        "    args: [mcp-server-fetch]\n"
        "    transport: stdio\n"
        "    keywords: [url, web]\n"
        "  - name: mydata\n"
        "    url: https://example.com/mcp\n"
        "    transport: streamable_http\n"
        "    trusted: false\n"
        "```\n",
        encoding="utf-8",
    )
    servers = cf.mcp_servers()
    assert len(servers) == 2
    names = {s["name"] for s in servers}
    assert names == {"fetch", "mydata"}


def test_mcp_registry_merges_into_client(root, monkeypatch):
    (root / "MCP.md").write_text(
        "```yaml\nservers:\n  - name: mydatasrv\n    command: uvx\n"
        "    args: [mcp-server-mydata]\n    keywords: [квазар]\n    trusted: true\n```\n",  # ЯВНЫЙ trusted (новый дефолт — без него не доверен)
        encoding="utf-8",
    )
    import src.data.mcp_client as mc
    # чистый старт реестра (идемпотентный флаг)
    monkeypatch.setattr(mc, "_registry_loaded", False)
    monkeypatch.setattr(mc, "TRUSTED_SERVERS", dict(mc.TRUSTED_SERVERS))
    monkeypatch.setattr(mc, "CATALOG", dict(mc.CATALOG))
    mc._load_user_registry()
    assert "mydatasrv" in mc.TRUSTED_SERVERS  # доверенный по умолчанию (юзер сам внёс)
    assert "mydatasrv" in mc.CATALOG
    assert mc.suggest_server("нужен квазар сейчас") == "mydatasrv"  # нашёлся по уникальному keyword


def test_bad_mcp_yaml_is_safe(root):
    (root / "MCP.md").write_text("просто текст без yaml-блока и без servers", encoding="utf-8")
    assert cf.mcp_servers() == []  # не падает, пусто
