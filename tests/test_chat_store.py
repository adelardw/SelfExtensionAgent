"""chat_store — постоянный индекс чатов/тредов (история, избранное, сжатие). Офлайн, temp-БД."""
import importlib
import os

import pytest


@pytest.fixture()
def cs(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CHATS_DB", str(tmp_path / "chats.db"))
    import src.chat_store as m
    importlib.reload(m)        # пере-инициализировать с временной БД
    m._conn = None
    yield m
    m._conn = None


def test_record_and_list(cs):
    cs.record_turn("t1", "local", "лучшие суши в Москве", "вот рейтинги: zoon, kp")
    cs.record_turn("t1", "local", "а адреса?", "по ссылкам")
    cs.record_turn("t2", "local", "включи музыку", "играет")
    rows = cs.list_threads("local")
    assert {r["thread_id"] for r in rows} == {"t1", "t2"}
    t1 = next(r for r in rows if r["thread_id"] == "t1")
    assert t1["title"].startswith("лучшие суши") and t1["msg_count"] == 4  # 2 обмена = 4 реплики


def test_title_from_first_message_only(cs):
    cs.record_turn("t", "local", "первый вопрос", "ответ")
    cs.record_turn("t", "local", "второй", "ответ2")
    assert cs.get_thread("t")["title"].startswith("первый вопрос")  # заголовок не перезатёрся


def test_favorites_toggle_and_filter(cs):
    cs.record_turn("a", "local", "qa", "aa")
    cs.record_turn("b", "local", "qb", "ab")
    assert cs.toggle_favorite("b") is True
    assert cs.toggle_favorite("b") is False
    cs.set_favorite("a", True)
    favs = cs.list_threads("local", favorites_only=True)
    assert [r["thread_id"] for r in favs] == ["a"]


def test_favorites_sort_first(cs):
    cs.record_turn("old", "local", "старый", "x")
    cs.record_turn("new", "local", "новый", "y")
    cs.set_favorite("old", True)  # избранный старый должен быть выше свежего обычного
    assert cs.list_threads("local")[0]["thread_id"] == "old"


def test_get_messages_full_and_tail(cs):
    for i in range(5):
        cs.record_turn("t", "local", f"q{i}", f"a{i}")
    assert len(cs.get_messages("t")) == 10
    tail = cs.get_messages("t", last=3)
    assert len(tail) == 3 and tail[-1]["content"] == "a4"  # хвост в хронологическом порядке


def test_summary_compression_flag(cs):
    cs.record_turn("t", "local", "q", "a")
    assert cs.list_threads("local")[0]["compressed"] == 0
    cs.set_summary("t", "сжатая память чата")
    assert cs.list_threads("local")[0]["compressed"] == 1
    assert cs.get_thread("t")["summary"] == "сжатая память чата"
    # полная история ПЕРЕЖИВАЕТ сжатие (саммари — индекс, не замена)
    assert len(cs.get_messages("t")) == 2


def test_rename_and_delete(cs):
    cs.record_turn("t", "local", "исходный", "a")
    cs.rename("t", "мой важный чат")
    assert cs.get_thread("t")["title"] == "мой важный чат"
    cs.delete_thread("t")
    assert cs.get_thread("t") is None and cs.get_messages("t") == []


def test_record_turn_robust_on_empty_thread_id(cs):
    cs.record_turn("", "local", "q", "a")  # не должно падать и не создаёт мусор
    assert cs.list_threads("local") == []
