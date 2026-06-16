"""
Видимость тихих деградаций (долг ревью #5): broad-except fallback'и зовут degradation.note(),
общий счётчик виден через snapshot()/total() и всплывает в /diagnose.
"""
from src import degradation


def test_note_counts_and_resets():
    degradation.reset()
    degradation.note("reflexion_failed", ValueError("x"))
    degradation.note("reflexion_failed")
    degradation.note("decompose_failed")
    snap = degradation.snapshot()
    assert snap == {"reflexion_failed": 2, "decompose_failed": 1}
    assert degradation.total() == 3
    degradation.reset()
    assert degradation.total() == 0 and degradation.snapshot() == {}


def test_diagnose_surfaces_degradations():
    from src.tracing.diagnose import diagnose

    degradation.reset()
    degradation.note("step_validation_skipped", RuntimeError("boom"))
    report = diagnose()                       # без memory_store — только трейс+деградации
    assert report["degradations"].get("step_validation_skipped") == 1
    assert any("деградаци" in f.lower() for f in report["findings"])
    degradation.reset()
