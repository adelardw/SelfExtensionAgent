"""sea doctor: классификация SearXNG по СОДЕРЖИМОМУ (не по HTTP-коду), рендер/exit-код,
офлайн-прогон всей батареи без сети. Offline."""
import io
import json
import urllib.request

import pytest

from src.maintenance import doctor as d


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_classify_searxng_not_set():
    assert d.classify_searxng("")[0] == "not_set"


def test_classify_searxng_down(monkeypatch):
    def boom(*a, **k):
        raise OSError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    st, detail = d.classify_searxng("http://stub:8080")
    assert st == "down" and "refused" in detail


def test_classify_searxng_alive_but_empty(monkeypatch):
    """Главная боль валидации: HTTP 200, но 0 результатов (апстримы в капче) — это FAIL,
    а не «работает»."""
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _Resp(json.dumps({"results": []}).encode()))
    st, detail = d.classify_searxng("http://stub:8080")
    assert st == "alive_empty" and "капч" in detail


def test_classify_searxng_ok(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _Resp(json.dumps({"results": [{"url": "x"}]}).encode()))
    assert d.classify_searxng("http://stub:8080")[0] == "ok"


def test_render_and_exit_semantics():
    results = [d._r("a", d.OK, "х"), d._r("b", d.FAIL, "y", "чини"), d._r("c", d.WARN, "z")]
    out = d.render(results)
    assert "❌ 1" in out and "⚠️ 1" in out and "↳ чини" in out
    assert any(r["status"] == d.FAIL for r in results)  # main() вернёт 1 при fail


def test_prune_checkpoints_keeps_latest_per_thread(tmp_path):
    """Прунер оставляет ПОСЛЕДНИЙ чекпоинт каждого треда (+его writes), остальное режет."""
    import sqlite3

    from src.maintenance.db_prune import prune_checkpoints

    db = tmp_path / "ck.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,"
              " parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, metadata BLOB)")
    c.execute("CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,"
              " task_id TEXT, idx INT, channel TEXT, type TEXT, value BLOB)")
    for t in ("t1", "t2"):
        for i in range(3):
            c.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
                      (t, "", f"0{i}", None, "x", b"d", b"m"))
            c.execute("INSERT INTO writes VALUES (?,?,?,?,?,?,?,?)",
                      (t, "", f"0{i}", "task", 0, "ch", "x", b"v"))
    c.commit(); c.close()

    r = prune_checkpoints(str(db), backup=True)
    assert r["status"] == "ok" and r["dropped_checkpoints"] == 4 and r["dropped_writes"] == 4
    assert (tmp_path / "ck.db.bak").exists()
    c = sqlite3.connect(db)
    rows = c.execute("SELECT thread_id, checkpoint_id FROM checkpoints ORDER BY thread_id").fetchall()
    assert rows == [("t1", "02"), ("t2", "02")]  # последний (MAX id) каждого треда жив
    assert c.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 2


def test_run_checks_offline_no_network():
    """Вся батарея в --offline: сетевые проверки скипаются, остальное не падает."""
    results = d.run_checks(offline=True)
    assert len(results) >= 10
    assert all(r["status"] in (d.OK, d.WARN, d.FAIL, d.SKIP) for r in results)
    skipped = {r["name"] for r in results if r["status"] == d.SKIP}
    assert "SearXNG" in skipped and "DDG-фолбэк" in skipped
    # песочница и load-check работают офлайн и должны быть зелёными
    by_name = {r["name"]: r for r in results}
    assert by_name["Песочница python"]["status"] == d.OK
    assert by_name["Load-check навыка (subprocess)"]["status"] == d.OK
