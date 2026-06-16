"""
Атомарная и сериализованная запись ParamStore (долг ревью improve #1 — гонка params.json).

Раньше add_fewshot/add_user_fewshot (из фоновых reflect-потоков + per-user воркера) писали голым
write_text без лока/atomic → гонка read-modify-write, а полузаписанный файл _load глотал в {} →
МОЛЧА обнулял ВСЕ выученные параметры. Тот же класс, что 2c. Тест воспроизводит конкуренцию.
"""
import json
import threading

from src.improve import prompt_store as ps


def test_concurrent_param_writes_never_corrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PARAMS_FILE", tmp_path / "params.json")
    errors: list[str] = []

    def worker(n: int) -> None:
        try:
            for i in range(40):
                data = ps._load()                       # read-modify-write под гонкой
                data[f"node{n}"] = {"fewshots": [{"q": f"q{i}", "a": "x"}]}
                ps._save(data)
                if ps.PARAMS_FILE.exists():
                    json.loads(ps.PARAMS_FILE.read_text(encoding="utf-8"))  # всегда ЦЕЛЫЙ JSON
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors[:3]
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())   # без .tmp-мусора
    json.loads(ps.PARAMS_FILE.read_text(encoding="utf-8"))           # финальный файл валиден


def test_skill_registry_atomic_under_concurrency(tmp_path):
    """#4: реестр навыков пишется атомарно+под локом (фон-reflect судит/удаляет навыки параллельно
    с main create/sync). Раньше голый write_text → read-modify-write гонка на одном JSON."""
    import json
    import threading
    from src.tools import skill_creation as sc

    reg = tmp_path / "registry.json"
    errors: list[str] = []

    def worker(n: int) -> None:
        try:
            for i in range(40):
                sc._save_reg_at(reg, {f"k{n}-{i}": {"has_tools": True}})
                if reg.exists():
                    json.loads(reg.read_text(encoding="utf-8"))   # всегда ЦЕЛЫЙ JSON
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors[:3]
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_user_fewshots_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "USER_FEWSHOTS_FILE", tmp_path / "user_fewshots.json")
    ps._save_users({"u1": {"reflexion": [{"q": "x", "a": "deliberate"}]}})
    assert json.loads((tmp_path / "user_fewshots.json").read_text())["u1"]
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())
