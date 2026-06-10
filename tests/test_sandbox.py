"""Песочница smoke-теста: подпроцесс, изоляция падений, жёсткий таймаут."""
import textwrap

from src import utils
from src.utils import run_tool_sandboxed


def _write_skill(tmp_path, body: str):
    f = tmp_path / "skill.py"
    f.write_text(textwrap.dedent(body), encoding="utf-8")
    return f


def test_sandbox_runs_tool(tmp_path):
    f = _write_skill(tmp_path, """
        from langchain_core.tools import tool

        @tool
        def greet(name: str) -> str:
            '''Здоровается.'''
            return f"привет, {name}!"
    """)
    ok, result = run_tool_sandboxed(f, "greet", {"name": "мир"})
    assert ok and "привет, мир" in result


def test_sandbox_isolates_crash(tmp_path):
    """Падение сгенерированного кода не роняет процесс агента."""
    f = _write_skill(tmp_path, """
        from langchain_core.tools import tool

        @tool
        def boom(x: str) -> str:
            '''Падает.'''
            raise RuntimeError("намеренный взрыв")
    """)
    ok, result = run_tool_sandboxed(f, "boom", {"x": "1"})
    assert not ok and "RuntimeError" in result


def test_sandbox_kills_hung_tool(tmp_path, monkeypatch):
    """Зависший код убивается по wall-таймауту вместе с процессом."""
    monkeypatch.setattr(utils, "SMOKE_IMPORT_GRACE", 8)
    f = _write_skill(tmp_path, """
        import time
        from langchain_core.tools import tool

        @tool
        def hang(x: str) -> str:
            '''Висит.'''
            time.sleep(120)
            return "never"
    """)
    ok, result = run_tool_sandboxed(f, "hang", {"x": "1"}, timeout=1)
    assert not ok and "завис" in result


def test_sandbox_missing_tool(tmp_path):
    f = _write_skill(tmp_path, """
        from langchain_core.tools import tool

        @tool
        def real(x: str) -> str:
            '''Есть.'''
            return x
    """)
    ok, result = run_tool_sandboxed(f, "nonexistent", {"x": "1"})
    assert not ok and "не найден" in result
