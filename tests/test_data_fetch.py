"""fetch_data — добыча датасета по URL → .xlsx (раздел B: закрывает «python_exec без сети»).

Парсинг (CSV с/без заголовка, JSON) и gunzip — детерминированно, БЕЗ сети. Полный flow — с моком
загрузки (`_download`) и без HITL (`full_auto`=True), артефакт пишется в tmp.
"""
import asyncio
import gzip
import os

from src.runtime import data_fetch as DF
from src.runtime import run_context as RC


def test_parse_csv_with_header():
    cols, rows = DF._parse_table("x.csv", b"Year,Temp\n2014,5.1\n2015,5.3\n")
    assert cols == ["Year", "Temp"]
    assert rows == [["2014", "5.1"], ["2015", "5.3"]]


def test_parse_csv_without_header():
    cols, rows = DF._parse_table("x.csv", b"2014,5.1\n2015,5.3\n")   # числовая 1-я строка → colN
    assert cols == ["col1", "col2"] and len(rows) == 2


def test_parse_json_list_of_dicts():
    cols, rows = DF._parse_table("x.json", b'[{"d":"2014-01-01","t":-5},{"d":"2014-01-02","t":-3}]')
    assert cols == ["d", "t"]
    assert rows[0] == ["2014-01-01", -5]


def test_maybe_gunzip():
    inner = b"a,b\n1,2\n"
    assert DF._maybe_gunzip("x.csv.gz", gzip.compress(inner)) == inner
    assert DF._maybe_gunzip("x.csv", inner) == inner               # не gzip → как есть


def test_fetch_builds_xlsx(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DF, "_download", lambda url: b"Year,Temp\n2014,5.1\n2015,5.3\n")  # без сети
    from src.runtime import hitl
    monkeypatch.setattr(hitl, "is_auto", lambda: True)                                      # auto-accept покрывает
    tool = DF.make_fetch_data_tool()
    with RC.request_scope("scope-fetch", "local"):
        out = asyncio.new_event_loop().run_until_complete(
            tool.ainvoke({"url": "http://x/y.csv", "filename": "temp"}))
        arts = RC.artifacts()
    assert "доставлен" in out and "2 строк" in out
    assert len(arts) == 1 and arts[0]["name"].endswith(".xlsx")
    assert os.path.exists(arts[0]["path"])
