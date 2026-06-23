"""Регрессия (живой eval поймал): ЛЮБОЙ файловый тул доставляет файл, а не пишет в cwd с «нельзя»."""
import os
import tempfile

import pytest

from src.runtime import run_context, artifacts


@pytest.fixture
def in_tmp():
    old = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    yield
    os.chdir(old)


def test_save_artifact_file_registers_and_contains(in_tmp):
    with run_context.request_scope("r", "local"):
        m = artifacts.save_artifact_file("notes.txt", "hello")
        assert run_context.artifacts()[0]["id"] == m["id"]
        p = artifacts.resolve_artifact(m["id"])
        assert p.exists() and p.read_text() == "hello"
        assert p.parent.name == m["id"]  # в контейнере artifacts/<id>/, не в cwd


def test_write_file_delivers_not_litters_cwd(in_tmp):
    from src.skills.file_operations.file_operations import write_file
    with run_context.request_scope("r", "local"):
        msg = write_file.invoke({"file_path": "data.csv", "content": "a,b\n1,2"})
        assert "ДОСТАВЛЕН" in msg  # тул сообщает агенту, что файл доставлен
        arts = run_context.artifacts()
        assert len(arts) == 1 and arts[0]["name"] == "data.csv"
        assert not os.path.exists("data.csv")  # НЕ в cwd (раньше сорил в репо)


def test_write_file_blocks_path_traversal(in_tmp):
    from src.skills.file_operations.file_operations import write_file
    with run_context.request_scope("r", "local"):
        write_file.invoke({"file_path": "../../etc/evil.csv", "content": "x"})
        m = run_context.artifacts()[0]
        assert "/" not in m["name"] and ".." not in m["name"]  # имя санитизировано


def test_wants_file_output_detector():
    """Embedding-сигнал «нужен файл-вывод» (исключает file-запросы из форса web_grounding→act)."""
    from src.graph.semantic_signals import wants_file_output as w
    # file-запросы (контент не размывает интент)
    assert w("Собери в один Excel-файл годовую инфляцию в России за 2019-2024")
    assert w("Выгрузи мои расходы в csv файл")
    # чистые факты/действия — НЕ файл (не должны ложно уходить из act-грундинга)
    assert not w("Какая была инфляция в России в 2023?")
    assert not w("Найди лучшие рестораны рядом")
