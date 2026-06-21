"""AST-гейт безопасности генерируемых навыков + защита core-скиллов + HITL."""
import asyncio

import pytest

from src.tools.utils_validation import validate_skill_code


def _bad(code: str) -> list[str]:
    ok, issues = validate_skill_code(code)
    assert not ok, f"должно быть отклонено: {code!r}"
    return issues


def test_catches_os_system():
    _bad("import os\nos.system('rm -rf /')")


def test_catches_from_import_alias():
    _bad("from os import system as s\ns('ls')")


def test_catches_subprocess_any_form():
    _bad("import subprocess\nsubprocess.run(['ls'])")
    _bad("import subprocess as sp\nsp.Popen(['ls'])")
    _bad("from subprocess import run\nrun(['ls'])")


def test_catches_getattr_bypass():
    # классический обход строкового матчинга: getattr(os, 'sys'+'tem')
    _bad("import os\nf = getattr(os, 'sys' + 'tem')\nf('ls')")


def test_catches_eval_exec_import():
    _bad("eval('1+1')")
    _bad("exec('x=1')")
    _bad("__import__('os').system('ls')")
    _bad("import importlib\nimportlib.import_module('subprocess')")


def test_catches_shutil_rmtree_and_reference():
    _bad("import shutil\nshutil.rmtree('/tmp/x')")
    _bad("import shutil\ncb = shutil.rmtree")  # передача без вызова


def test_allows_clean_skill_code():
    code = (
        "import json\n"
        "import urllib.request\n"
        "from pathlib import Path\n"
        "from langchain_core.tools import tool\n\n"
        "@tool\n"
        "def fetch_json(url: str) -> str:\n"
        "    '''Скачивает JSON.'''\n"
        "    with urllib.request.urlopen(url, timeout=10) as r:\n"
        "        return json.dumps(json.load(r))[:500]\n"
    )
    ok, issues = validate_skill_code(code)
    assert ok, issues


def test_syntax_error_rejected():
    ok, issues = validate_skill_code("def broken(:\n  pass")
    assert not ok and "SyntaxError" in issues[0]


# ── защита core-навыков: force недоступен LLM ──────────────────────────

@pytest.fixture()
def tmp_skills(tmp_path, monkeypatch):
    from src.tools import skill_creation as sc

    monkeypatch.setattr(sc, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(sc, "REGISTRY_FILE", tmp_path / "registry.json")
    monkeypatch.setattr(sc, "PROTECTED_SKILLS", set())
    return sc


def test_protected_skill_not_deletable_by_tool(tmp_skills):
    sc = tmp_skills
    (sc.SKILLS_DIR / "core_x").mkdir(parents=True)
    sc._save_registry({"core_x": {"description": "", "protected": True}})

    msg = sc.delete_skill.invoke({"name": "core_x"})
    assert "PROTECTED" in msg and (sc.SKILLS_DIR / "core_x").exists()
    # параметра force у тула больше нет вообще
    assert "force" not in (sc.delete_skill.args_schema.model_json_schema().get("properties", {}))

    # владельческое удаление (не tool) работает
    msg = sc.force_delete_skill("core_x")
    assert "deleted" in msg and not (sc.SKILLS_DIR / "core_x").exists()


def test_create_skill_security_gate(tmp_skills, monkeypatch):
    sc = tmp_skills
    monkeypatch.delenv("AGENT_ALLOW_RISKY_SKILLS", raising=False)
    msg = sc.create_skill.invoke({
        "name": "evil",
        "description": "x",
        "tool_code": "import os\nfrom langchain_core.tools import tool\n@tool\ndef go(c: str) -> str:\n    '''x'''\n    os.system(c)\n    return 'ok'\n",
    })
    assert msg.startswith("Rejected") and not (sc.SKILLS_DIR / "evil" / "evil.py").exists()


# ── human-in-the-loop обёртка ──────────────────────────────────────────

def test_hitl_deny_by_default_and_allow_with_confirmer():
    from langchain_core.tools import tool

    from src.runtime import hitl

    @tool
    def open_app(name: str) -> str:
        """Открывает приложение."""
        return f"opened {name}"

    wrapped = hitl.wrap_with_confirmation(open_app, "device_control")

    hitl.set_confirmer(None)  # нет канала подтверждения → deny by default
    res = asyncio.run(wrapped.ainvoke({"name": "FaceTime"}))
    assert "отклонено" in res.lower()

    hitl.set_confirmer(lambda desc: True)
    try:
        res = asyncio.run(wrapped.ainvoke({"name": "FaceTime"}))
        assert res == "opened FaceTime"
    finally:
        hitl.set_confirmer(None)
