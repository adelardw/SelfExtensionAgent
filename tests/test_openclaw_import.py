"""Импорт OpenClaw-скиллов: парсинг SKILL.md, allowlist бинарников, HITL, безопасность."""
import asyncio

import pytest

from src.tools import openclaw_import as oci

SKILL_MD = """---
name: weather-cli
description: "Current weather via wttr.in curl."
homepage: https://wttr.in
metadata:
  {
    "openclaw":
      {
        "emoji": "☔",
        "os": ["darwin", "linux"],
        "install":
          [
            { "id": "brew", "kind": "brew", "formula": "curl", "bins": ["curl"], "label": "Install curl" }
          ]
      }
  }
---

# Weather
Use `curl wttr.in/<city>?format=3` for a quick forecast.
"""


def test_parse_skill_md():
    meta, body = oci.parse_skill_md(SKILL_MD)
    assert meta["name"] == "weather-cli"
    assert "curl wttr.in" in body
    oc = meta["metadata"]["openclaw"]
    assert oci._collect_bins(oc) == ["curl"]  # из install[].bins


def test_collect_bins_union():
    oc = {"requires": {"bins": ["a"]}, "install": [{"bins": ["b"]}, {"bins": ["a", "c"]}]}
    assert oci._collect_bins(oc) == ["a", "b", "c"]  # union без дублей, с порядком


def test_sanitize():
    assert oci._sanitize("weather-cli") == "weather_cli"
    assert oci._sanitize("Apple Notes!") == "apple_notes"


@pytest.fixture()
def tmp_skills(tmp_path, monkeypatch):
    from src.tools import skill_creation as sc

    monkeypatch.setattr(sc, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(sc, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(sc, "PROTECTED_SKILLS", set())
    monkeypatch.setattr(oci, "SKILLS_DIR", tmp_path)
    return tmp_path


def _make_source(tmp_path):
    src = tmp_path / "src_skill"
    src.mkdir()
    (src / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return str(src)


def test_import_creates_skill_and_registry(tmp_skills):
    from src.tools import skill_creation as sc

    report = oci.import_openclaw_skill(_make_source(tmp_skills))
    assert "weather_cli" in report and "curl" in report

    reg = sc._load_registry()
    assert reg["weather_cli"]["imported"] is True
    d = sc.SKILLS_DIR / "weather_cli"
    assert (d / "SKILL.md").exists() and (d / "weather_cli.py").exists() and (d / "prompt.md").exists()


def test_imported_skill_under_hitl(tmp_skills):
    from src.runtime import hitl
    from src.tools import skill_creation as sc

    oci.import_openclaw_skill(_make_source(tmp_skills))
    # импортированный навык требует подтверждения, даже если его нет в config.confirm
    monkeypatch_require(hitl, True)
    assert hitl.needs_confirmation("weather_cli") is True


def test_wrapper_allowlist_blocks_foreign_bin(tmp_skills, monkeypatch):
    import os

    from src.runtime import hitl
    from src.tools import skill_creation as sc

    monkeypatch.setenv("AGENT_DRY_RUN", "1")
    oci.import_openclaw_skill(_make_source(tmp_skills))
    tools = {t.name: t for t in sc.get_all_loaded_skill_tools(["weather_cli"])}
    assert set(tools) == {"weather_cli_instructions", "weather_cli_run"}

    hitl.set_confirmer(lambda d: True)
    try:
        ok = asyncio.run(tools["weather_cli_run"].ainvoke({"command": "curl wttr.in/Almaty?format=3"}))
        assert ok.startswith("[dry-run]")
        blocked = asyncio.run(tools["weather_cli_run"].ainvoke({"command": "rm -rf /"}))
        assert "не разрешён" in blocked
    finally:
        hitl.set_confirmer(None)


def monkeypatch_require(hitl_mod, value: bool) -> None:
    hitl_mod.REQUIRE_CONFIRMATION = value
