"""Навык `code` (L2): read-тулзы (glob/grep/tree/read) работают; edit_file точечная; run_bash
с dry-run. КРИТИЧНО (требование юзера): run_bash зависит от work_mode через HITL — plan блокирует,
auto выполняет, read-тулзы (glob) свободно. Offline."""
import asyncio
import importlib.util
from pathlib import Path

import pytest

# Грузим навык напрямую (как делает skill-loader через exec_module).
_spec = importlib.util.spec_from_file_location("skills.code", "src/skills/code/code.py")
code = importlib.util.module_from_spec(_spec)
import sys
sys.modules["skills.code"] = code
_spec.loader.exec_module(code)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # Навык `code` скоупится к корню проекта (AGENT_PROJECT_ROOT/cwd) — анти-эксфильтрация
    # (read-тулзы readonly, без HITL). Штатно навести его на ДРУГОЙ проект = указать
    # AGENT_PROJECT_ROOT (как в проде), что фикстура и делает для временного репо.
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def test_glob_and_grep_and_tree(repo):
    g = code.glob_files.invoke({"pattern": "**/*.py", "path": str(repo)})
    assert "a.py" in g and "b.py" in g
    gr = code.grep_repo.invoke({"pattern": "def foo", "path": str(repo)})
    assert "a.py" in gr and "def foo" in gr
    tr = code.list_tree.invoke({"path": str(repo)})
    assert "pkg/" in tr


def test_code_tools_scoped_and_secrets_blocked(repo, tmp_path):
    """Анти-эксфильтрация (баг ревью): read/edit-тулзы навыка `code` не выходят за корень
    проекта и не отдают секрет-файлы (.env/ключи). read-тулзы readonly → без HITL, поэтому
    скоуп+денилист — единственный барьер от утечки секретов в контекст и далее на внешний хост."""
    (repo / ".env").write_text("OPEN_ROUTER_API_KEY=sk-secret-xyz\n", encoding="utf-8")
    # секрет-файл В корне — не читается ни read_lines, ни grep, ни glob/tree его не светят
    assert "запрещён" in code.read_lines.invoke({"file_path": str(repo / ".env")})
    assert "sk-secret-xyz" not in code.grep_repo.invoke({"pattern": "sk-secret-xyz", "path": str(repo)})
    assert ".env" not in code.glob_files.invoke({"pattern": ".env", "path": str(repo)})
    assert ".env" not in code.list_tree.invoke({"path": str(repo)})
    # путь ВНЕ корня проекта — отклонён (и абсолютный, и через traversal)
    assert "вне проекта" in code.read_lines.invoke({"file_path": "/etc/hosts"})
    assert "вне проекта" in code.read_lines.invoke({"file_path": str(tmp_path.parent / "x")})
    # правка секрет-файла — отказ (write-путь тоже защищён)
    assert "запрещена" in code.edit_file.invoke(
        {"file_path": str(repo / ".env"), "old_string": "x", "new_string": "y"})
    # легитимный код В корне — по-прежнему читается/правится
    assert "def foo" in code.read_lines.invoke({"file_path": str(repo / "pkg" / "a.py")})


def test_read_lines_numbered(repo):
    r = code.read_lines.invoke({"file_path": str(repo / "pkg" / "a.py"), "start": 1, "end": 2})
    assert "1\t" in r and "def foo" in r


def test_edit_file_surgical(repo):
    f = repo / "pkg" / "a.py"
    ok = code.edit_file.invoke({"file_path": str(f), "old_string": "return 42", "new_string": "return 43"})
    assert "OK" in ok and "return 43" in f.read_text()
    # отсутствующий фрагмент → отказ, не падение
    bad = code.edit_file.invoke({"file_path": str(f), "old_string": "НЕТ_ТАКОГО", "new_string": "x"})
    assert "не найден" in bad


def test_run_bash_dry_run(monkeypatch):
    monkeypatch.setenv("AGENT_DRY_RUN", "1")
    r = code.run_bash.invoke({"command": "echo привет"})
    assert "dry-run" in r and "echo привет" in r


@pytest.mark.asyncio
async def test_run_bash_respects_work_mode_via_hitl(repo, monkeypatch):
    """ТРЕБОВАНИЕ ЮЗЕРА: bash — с разрешением и зависит от мода. Через hitl.wrap_with_confirmation."""
    monkeypatch.delenv("AGENT_DRY_RUN", raising=False)
    import src.runtime.hitl as hitl
    wrapped_bash = hitl.wrap_with_confirmation(code.run_bash, "code")
    wrapped_glob = hitl.wrap_with_confirmation(code.glob_files, "code")

    # plan-режим: side-effect bash НЕ исполняется (стаб [PLAN], не вывод команды)
    hitl.set_work_mode("plan")
    out = await wrapped_bash.ainvoke({"command": "echo PLANMARK"})
    assert "[PLAN]" in out and "НЕ исполнен" in out  # заблокировано, реального вывода нет
    # read-only glob в plan — свободно работает (исследовать можно)
    g = await wrapped_glob.ainvoke({"pattern": "**/*.py", "path": str(repo)})
    assert "a.py" in g

    # auto-режим: bash выполняется без вопроса
    hitl.set_work_mode("auto")
    out2 = await wrapped_bash.ainvoke({"command": "echo RAN_OK"})
    assert "RAN_OK" in out2
    hitl.set_work_mode("manual")  # вернуть дефолт
