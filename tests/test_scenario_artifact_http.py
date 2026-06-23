"""СЦЕНАРНАЯ валидация доставки файла: тул производит .xlsx → сервер реально отдаёт его по
GET /artifact/{id} настоящими байтами; traversal/мусор → 404. (Главная боль: «дай excel».)"""
import os
import tempfile

import pytest


@pytest.fixture
def client_and_artifacts():
    # импорт сервера из КОРНЯ репо (config.yml есть), артефакты — в temp (write+read один cwd)
    from fastapi.testclient import TestClient
    from src.interface.server import app
    from src.runtime import artifacts, run_context
    old = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    yield TestClient(app), artifacts, run_context
    os.chdir(old)


def test_scenario_export_then_download(client_and_artifacts):
    client, artifacts, run_context = client_and_artifacts
    with run_context.request_scope("scn-http", "local"):
        tool = artifacts.make_export_tool()
        tool.invoke({"filename": "rf_market", "columns": ["Год", "ВВП_изм_%"],
                     "rows": [["2023", "3.6"], ["2024", "4.1"]], "sheet": "Макро"})
        meta = run_context.artifacts()[0]
    aid = meta["id"]

    r = client.get(f"/artifact/{aid}")
    assert r.status_code == 200
    assert r.content[:2] == b"PK"                       # реальные xlsx-байты (zip-сигнатура)
    assert "attachment" in r.headers.get("content-disposition", "")
    assert "rf_market.xlsx" in r.headers.get("content-disposition", "")


def test_scenario_bad_ids_404(client_and_artifacts):
    client, _, _ = client_and_artifacts
    assert client.get("/artifact/deadbeefdead").status_code == 404      # hex, но нет такого
    assert client.get("/artifact/not_hex_id").status_code == 404        # не hex
    assert client.get("/artifact/..%2f..%2fetc%2fpasswd").status_code == 404  # traversal
