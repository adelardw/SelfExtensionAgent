"""Артефакты-для-отдачи: export_table пишет скачиваемый .xlsx, регистрирует в run_context,
сервер резолвит по id (анти-traversal), тул всегда доступен на шаге."""
import os
import tempfile

import pytest

from src.runtime import artifacts, run_context


@pytest.fixture
def in_tmp_cwd():
    old = os.getcwd()
    d = tempfile.mkdtemp()
    os.chdir(d)
    yield d
    os.chdir(old)


def test_write_table_produces_xlsx_and_registers(in_tmp_cwd):
    with run_context.request_scope("r1", "local"):
        m = artifacts.write_table("report", ["A", "B"], [["1", "2"], ["3", "4"]], "S1")
        assert m["kind"] == "xlsx" and m["nrows"] == 2
        assert m["name"].endswith(".xlsx")
        reg = run_context.artifacts()
        assert len(reg) == 1 and reg[0]["id"] == m["id"]
        p = artifacts.resolve_artifact(m["id"])
        assert p is not None and p.exists() and p.stat().st_size > 0


def test_same_filename_adds_sheet_not_new_file(in_tmp_cwd):
    import openpyxl
    with run_context.request_scope("r2", "local"):
        artifacts.write_table("multi", ["A"], [["x"]], "First")
        m2 = artifacts.write_table("multi", ["B"], [["y"]], "Second")
        # один артефакт (та же книга), два листа
        assert len(run_context.artifacts()) == 1
        wb = openpyxl.load_workbook(artifacts.resolve_artifact(m2["id"]))
        assert wb.sheetnames == ["First", "Second"]


def test_resolve_artifact_blocks_traversal(in_tmp_cwd):
    assert artifacts.resolve_artifact("../etc/passwd") is None
    assert artifacts.resolve_artifact("../../secret") is None
    assert artifacts.resolve_artifact("not-hex-id") is None
    assert artifacts.resolve_artifact("") is None
    assert artifacts.resolve_artifact("deadbeef") is None  # hex, но нет такого каталога


def test_filename_sanitized(in_tmp_cwd):
    with run_context.request_scope("r3", "local"):
        m = artifacts.write_table("../../evil name!.xlsx", ["A"], [["1"]])
        assert "/" not in m["name"] and ".." not in m["name"]
        assert artifacts.resolve_artifact(m["id"]).parent.name == m["id"]


def test_export_tool_always_on_step_tools():
    """export_table — всегда-доступный core-тул шага (как python_exec/current_datetime)."""
    tool = artifacts.make_export_tool()
    assert tool.name == "export_table"
    # в safe-tools (экспорт своих данных не таинтит гейт python_exec)
    from src.graph.agent import _INTERNAL_SAFE_TOOLS
    assert "export_table" in _INTERNAL_SAFE_TOOLS


def test_artifacts_scoped_per_run(in_tmp_cwd):
    with run_context.request_scope("runA", "local"):
        artifacts.write_table("a", ["x"], [["1"]])
        assert len(run_context.artifacts()) == 1
    # другой прогон — чистый реестр (cleanup по выходе из scope)
    with run_context.request_scope("runB", "local"):
        assert run_context.artifacts() == []
