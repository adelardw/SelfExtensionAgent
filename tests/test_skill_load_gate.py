"""
Гейт ПЕРЕД exec_module (закрытие дыры «exec до HITL») + кэш модулей навыков.

Покрывает решение-таблицу доверия при загрузке навыка:
  • core/protected      → exec без AST-гейта (автор доверен);
  • imported (OpenClaw)  → module-level гейт (subprocess в теле ок, при импорте — нет);
  • прочее (orphan)      → полный AST-гейт (контракт «чистый stdlib»);
а также: модуль не ре-exec'ится на каждом шаге (кэш по mtime).
"""
import os
import time
from pathlib import Path

import pytest

from src.tools import skill_creation as sc
from src.utils_validation import validate_module_level


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SKILLS_DIR", str(tmp_path))
    sc._MODULE_CACHE.clear()
    return tmp_path


def _write_skill(root: Path, name: str, body: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    py = d / f"{name}.py"
    py.write_text(body, encoding="utf-8")
    return py


_TOOL = (
    "from langchain_core.tools import tool\n"
    "@tool\n"
    "def ping() -> str:\n"
    '    """p"""\n'
    "    return 'pong'\n"
)


def test_trusted_core_skill_bypasses_gate():
    assert sc._trusted_skill("web_search") is True
    assert sc._trusted_skill("code") is True            # core, legit subprocess
    assert sc._trusted_skill("__nonexistent__") is False


def test_orphan_with_subprocess_is_refused(skills_root):
    # навык НЕ в реестре (orphan на диске), не imported → полный гейт
    py = _write_skill(skills_root, "orphan", "import subprocess\n" + _TOOL)
    mod, reason = sc._load_skill_module("orphan", py)
    assert mod is None
    assert "subprocess" in reason


def test_module_level_danger_refused(skills_root):
    py = _write_skill(skills_root, "evil", "import subprocess\nsubprocess.run(['echo','x'])\n")
    mod, reason = sc._load_skill_module("evil", py)
    assert mod is None


def test_module_cache_avoids_reexec(skills_root, monkeypatch):
    # доверенный навык: грузится и кэшируется; повторная загрузка — тот же объект без ре-exec
    py = _write_skill(skills_root, "cached", _TOOL)
    monkeypatch.setattr(sc, "PROTECTED_SKILLS", sc.PROTECTED_SKILLS | {"cached"})
    m1, r1 = sc._load_skill_module("cached", py)
    m2, r2 = sc._load_skill_module("cached", py)
    assert m1 is not None and m1 is m2
    assert r2 == "OK (cached)"


def test_cache_invalidates_on_mtime_change(skills_root, monkeypatch):
    py = _write_skill(skills_root, "edited", _TOOL)
    monkeypatch.setattr(sc, "PROTECTED_SKILLS", sc.PROTECTED_SKILLS | {"edited"})
    m1, _ = sc._load_skill_module("edited", py)
    time.sleep(0.01)
    os.utime(py, (time.time() + 5, time.time() + 5))     # сдвигаем mtime
    m2, r2 = sc._load_skill_module("edited", py)
    assert r2 == "OK"                                    # перезагрузка, не кэш
    assert m1 is not m2


def test_module_level_gate_allows_subprocess_in_body():
    # модель OpenClaw-обёртки: subprocess в ТЕЛЕ тула (под HITL+allowlist) — допустим
    ok, _ = validate_module_level(
        "import subprocess\ndef run(c):\n    return subprocess.run(c)\n"
    )
    assert ok is True


def test_module_level_gate_blocks_aliased_import_call():
    ok, issues = validate_module_level("import subprocess as sp\nsp.Popen(['x'])\n")
    assert ok is False and issues


# ── #2: рантайм-вызов недоверенного навыка идёт в subprocess-песочницу ─────────
def test_should_sandbox_policy():
    assert sc._should_sandbox("generated_x", {}) is True            # сгенерированный → песочница
    assert sc._should_sandbox("web_search", {}) is False            # core/protected → in-process
    assert sc._should_sandbox("openclaw_x", {"imported": True}) is False  # imported → своя модель


def test_sandbox_wrap_runs_in_subprocess(skills_root):
    import asyncio
    import os as _os

    py = _write_skill(
        skills_root, "pidskill",
        "import os\nfrom langchain_core.tools import tool\n"
        "@tool\ndef whoami() -> str:\n    \"\"\"pid\"\"\"\n    return f'pid={os.getpid()}'\n",
    )
    from langchain_core.tools import tool

    @tool
    def whoami() -> str:
        """pid"""
        return "in-process"

    wrapped = sc._sandbox_wrap(whoami, "pidskill", py)
    res = asyncio.run(wrapped.ainvoke({}))
    assert res.startswith("pid=")                       # реально выполнился (sandbox)
    assert res != f"pid={_os.getpid()}"                 # в ДРУГОМ процессе, не в агенте
