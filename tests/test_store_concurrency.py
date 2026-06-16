"""
Конкурентность стора и атомарность кодбука (долги ревью 2b/2c).

Раньше: один shared sqlite-conn на процесс → фоновый reflect-поток пишет параллельно с основным
→ «database is locked»/переплетённые транзакции. И `intent._save()` неатомарно перезаписывал
весь JSON → гонка фоновых потоков била файл. Эти тесты воспроизводят именно конкурентный сценарий.
"""
import json
import tempfile
import threading
from pathlib import Path

import pytest


def test_store_wal_enabled():
    from src.memory.store import MemoryStore

    db = str(Path(tempfile.mkdtemp()) / "m.db")
    s = MemoryStore(db)
    try:
        assert s._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        s.close()


def test_concurrent_writers_no_lock_errors():
    """8 потоков пишут эпизоды/факты/рефлексии параллельно — ни одной ошибки sqlite."""
    from src.memory.store import MemoryStore

    db = str(Path(tempfile.mkdtemp()) / "m.db")
    s = MemoryStore(db)
    errors: list[str] = []

    def worker(n: int) -> None:
        try:
            for i in range(40):
                s.add_episode("u", f"q{n}-{i}", "a", outcome="ok")
                s.add_fact("u", f"k{n}-{i}", "v", importance=0.5)
                s.add_reflection("u", f"r{n}-{i}")
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors[:3]
    assert s.episode_count("u") == 8 * 40
    s.close()


def test_thread_local_close_does_not_affect_other_threads():
    """close() в фоновом reflect-потоке закрывает ТОЛЬКО своё соединение (thread-local) —
    основной поток продолжает писать. Пинит инвариант мелочи #3 (явный close эфемерного conn)."""
    from src.memory.store import MemoryStore

    s = MemoryStore(str(Path(tempfile.mkdtemp()) / "m.db"))
    try:
        s.add_episode("u", "main", "a")

        def bg() -> None:
            s.add_episode("u", "bg", "a")   # фоновый поток получает своё соединение
            s.close()                       # закрывает только его

        t = threading.Thread(target=bg)
        t.start()
        t.join()

        s.add_episode("u", "main2", "a")    # основной conn жив после close() в фоне
        assert s.episode_count("u") == 3
    finally:
        s.close()


def test_intent_save_atomic_under_concurrency(monkeypatch):
    """Параллельные _save из фоновых потоков: читатель всегда видит ЦЕЛЫЙ JSON, без .tmp-мусора."""
    import src.intent as intent

    d = Path(tempfile.mkdtemp())
    monkeypatch.setattr(intent, "CODEBOOK_FILE", d / "cb.json")
    router = intent.IntentRouter()  # напрямую, не shared-синглтон (его мог подменить др. тест)
    errors: list[str] = []

    def worker(n: int) -> None:
        try:
            for i in range(50):
                router._entries = [{"label": f"l{n}", "text": f"t{i}", "vec": [0.1, 0.2]}]
                router._save()
                if intent.CODEBOOK_FILE.exists():
                    json.loads(intent.CODEBOOK_FILE.read_text(encoding="utf-8"))  # должен парситься всегда
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors[:3]
    assert not any(p.suffix == ".tmp" for p in d.iterdir())  # временные файлы убраны
