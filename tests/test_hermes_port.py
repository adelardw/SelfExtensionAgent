"""Hermes-порт: (1) load-check навыка в подпроцессе (не в хосте), (2) usage-статистика +
куратор реестра, (3) триггер ретроспективной дистилляции (предикат + счётчик тул-вызовов),
(4) frontmatter SKILL.md (agentskills.io), (5) гибридный ранкер BM25+эмбеддинги (RRF).
Offline: без LLM и без сети (эмбеддер — стаб)."""
import os

import pytest

import src.tools.skill_creation as sc


_TOOLCODE = ("from langchain_core.tools import tool\n\n@tool\ndef add_one(x: int) -> int:\n"
             "    '''Прибавить 1.'''\n    return x + 1\n")


@pytest.fixture
def skl(tmp_path, monkeypatch):
    """Изолированный ГЛОБАЛЬНЫЙ ярус навыков во временном каталоге. Снимаем оба гейта
    record_skill_usage (eval/pytest) — здесь реестр временный, статистику писать МОЖНО."""
    monkeypatch.delenv("AGENT_EVAL_MODE", raising=False)
    monkeypatch.setenv("AGENT_USAGE_TRACKING_IN_TESTS", "1")  # реестр тут временный — писать можно
    monkeypatch.setattr(sc, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(sc, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _mk_skill(base, name, py_body, md="описание"):
    d = base / name
    d.mkdir(parents=True)
    (d / f"{name}.py").write_text(py_body, encoding="utf-8")
    (d / f"{name}.md").write_text(md, encoding="utf-8")


# ── 1. изоляция исполнения: load-check идёт в ПОДПРОЦЕССЕ ──────────────────────

def test_loadcheck_ok_and_isolated(skl):
    """Загружаемый навык проходит; module-level код НЕ исполняется в процессе агента."""
    from src.utils import _skill_loadable

    _mk_skill(skl, "iso_probe",
              "import os\nos.environ['HERMES_ISO_MARKER'] = '1'\n" + _TOOLCODE)
    sc._load_registry()  # ensure dirs
    ok, msg = _skill_loadable("iso_probe")
    assert ok and "add_one" in msg
    # маркер выставлен в ПОДПРОЦЕССЕ → окружение хоста чистое (раньше exec шёл in-process)
    assert "HERMES_ISO_MARKER" not in os.environ


def test_loadcheck_broken_skill(skl):
    from src.utils import _skill_loadable

    _mk_skill(skl, "broken_probe", "import nonexistent_module_xyz\n" + _TOOLCODE)
    ok, msg = _skill_loadable("broken_probe")
    assert not ok and "nonexistent_module_xyz" in msg


def test_loadcheck_no_tools(skl):
    from src.utils import _skill_loadable

    _mk_skill(skl, "empty_probe", "X = 1\n")
    ok, msg = _skill_loadable("empty_probe")
    assert not ok and "@tool" in msg


# ── 2. жизненный цикл: usage-статистика + куратор ──────────────────────────────

def test_usage_stats_recorded(skl):
    sc.create_skill.invoke({"name": "stat_skill", "description": "тест", "scope": "global"})
    sc.record_skill_usage(["stat_skill"], win=True)
    sc.record_skill_usage(["stat_skill"], win=False)
    meta = sc._load_registry()["stat_skill"]
    assert meta["uses"] == 2 and meta["wins"] == 1 and meta["last_used_at"]


def test_usage_not_recorded_in_eval(skl, monkeypatch):
    sc.create_skill.invoke({"name": "eval_skill", "description": "тест", "scope": "global"})
    monkeypatch.setenv("AGENT_EVAL_MODE", "1")
    sc.record_skill_usage(["eval_skill"], win=True)
    assert sc._load_registry()["eval_skill"].get("uses", 0) == 0


def test_win_accepts_temporary_skill(skl):
    """Temporary-навык (в т.ч. дистиллированный), победивший в реальном прогоне, принят."""
    sc.create_skill.invoke({"name": "temp_win", "description": "тест", "scope": "global"})
    sc.mark_temporary("temp_win")
    assert sc._load_registry()["temp_win"].get("temporary")
    sc.record_skill_usage(["temp_win"], win=True)
    assert not sc._load_registry()["temp_win"].get("temporary")


def test_curator_prunes_systematic_loser(skl):
    sc.create_skill.invoke({"name": "loser", "description": "тест", "scope": "global"})
    for _ in range(6):
        sc.record_skill_usage(["loser"], win=False)
    report = sc.sync_registry()
    assert "loser" in report["curated_out"]
    assert "loser" not in sc._load_registry()


def test_curator_keeps_winner_and_protected(skl, monkeypatch):
    sc.create_skill.invoke({"name": "winner", "description": "тест", "scope": "global"})
    for _ in range(6):
        sc.record_skill_usage(["winner"], win=True)
    sc.create_skill.invoke({"name": "core_like", "description": "тест", "scope": "global"})
    reg = sc._load_registry()
    reg["core_like"].update({"protected": True, "uses": 10, "wins": 0})
    sc._save_registry(reg)
    report = sc.sync_registry()
    assert report["curated_out"] == []
    reg = sc._load_registry()
    assert "winner" in reg and "core_like" in reg


# ── 3. ретроспективная дистилляция: предикат + счётчик тул-вызовов ─────────────

def test_distill_worthy_predicate():
    from src.graph.agent import _distill_worthy

    assert _distill_worthy("ok", "heavy", "", 6, False)
    assert _distill_worthy("ok", "deliberate", "", 5, False)
    assert not _distill_worthy("low_conf", "heavy", "", 6, False)   # неуспех
    assert not _distill_worthy("ok", "fast", "", 6, False)          # дешёвый режим
    assert not _distill_worthy("ok", "heavy", "existing", 6, False)  # навык уже создан
    assert not _distill_worthy("ok", "heavy", "", 4, False)         # мало тул-вызовов
    assert not _distill_worthy("ok", "heavy", "", 6, True)          # негативная реакция


def test_tool_call_counter_scoped_and_cleaned():
    from src.runtime import run_context as rc

    with rc.request_scope("run_distill_test", "u1"):
        rc.note_tool_calls(3)
        rc.note_tool_calls(2)
        assert rc.tool_calls_count() == 5
    with rc.request_scope("run_distill_test", "u1"):
        assert rc.tool_calls_count() == 0  # cleanup по выходе из scope


# ── 4. frontmatter SKILL.md ────────────────────────────────────────────────────

def test_create_skill_writes_frontmatter(skl):
    sc.create_skill.invoke({
        "name": "fm_skill", "description": "# FM\nтело описания", "scope": "global",
        "when_to_use": "когда нужен тест frontmatter"})
    raw = (skl / "fm_skill" / "fm_skill.md").read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    fm, body = sc.split_frontmatter(raw)
    assert fm["name"] == "fm_skill" and fm["when_to_use"] == "когда нужен тест frontmatter"
    assert "тело описания" in body and not body.startswith("---")
    assert sc._load_registry()["fm_skill"]["when_to_use"] == "когда нужен тест frontmatter"


def test_frontmatter_not_duplicated(skl):
    ready = "---\nname: pre\n---\n\nготовое описание"
    sc.create_skill.invoke({"name": "pre", "description": ready, "scope": "global"})
    raw = (skl / "pre" / "pre.md").read_text(encoding="utf-8")
    assert raw.count("---\n") == 2  # один блок frontmatter, не два


def test_prompt_injection_strips_frontmatter(skl):
    sc.create_skill.invoke({
        "name": "fm_view", "description": "тело для промпта", "scope": "global",
        "when_to_use": "триггер-фраза селектора"})
    out = sc.get_skills_for_prompt.invoke({})
    assert "тело для промпта" in out
    assert "Когда использовать: триггер-фраза селектора" in out
    assert "---\nname:" not in out  # YAML в промпт не течёт


def test_old_md_without_frontmatter_still_works(skl):
    _mk_skill(skl, "legacy", _TOOLCODE, md="старое описание без frontmatter")
    report = sc.sync_registry()
    assert "legacy" in report["added"]
    assert sc._load_registry()["legacy"]["description"].startswith("старое описание")


# ── 5. гибридный ранкер (RRF) ──────────────────────────────────────────────────

class _StubEmbedder:
    enabled = True

    def embed(self, text: str):
        # детерминированная «семантика»: погодные тексты → ось 0, табличные → ось 1
        t = text.lower()
        if "погод" in t or "дожд" in t:
            return [1.0, 0.0]
        if "таблиц" in t or "csv" in t:
            return [0.0, 1.0]
        return [0.5, 0.5]


@pytest.fixture
def stub_emb(skl, monkeypatch):
    monkeypatch.setattr(sc, "_skill_embedder_inst", _StubEmbedder())
    _mk_skill(skl, "weather_now", _TOOLCODE, md="погода прогноз")
    _mk_skill(skl, "csv_tool", _TOOLCODE, md="таблица csv агрегats")
    return skl


def test_rank_semantic_wins_without_lexical_overlap(stub_emb):
    names = ["weather_now", "csv_tool"]
    docs = ["weather_now погода прогноз", "csv_tool таблица csv"]
    # запрос без лексического пересечения — BM25 пуст, спасает эмбеддинг-ось
    idx = sc.rank_skill_docs(docs, "дождь завтра", 2, names=names)
    assert idx and idx[0] == 0
    assert (stub_emb / ".emb_cache.json").exists()  # кэш векторов лёг на диск


def test_rank_falls_back_to_bm25_when_disabled(skl, monkeypatch):
    class _Null:
        enabled = False

        def embed(self, text):
            return None

    monkeypatch.setattr(sc, "_skill_embedder_inst", _Null())
    docs = ["alpha уникальное слово совпадение", "beta другое"]
    idx = sc.rank_skill_docs(docs, "уникальное слово", 2, names=["a", "b"])
    assert idx == [0]


def test_rank_rrf_merges_both_signals(stub_emb):
    names = ["weather_now", "csv_tool"]
    docs = ["weather_now погода прогноз", "csv_tool таблица csv"]
    # лексика и семантика согласны → csv первым, RRF не ломает согласие
    idx = sc.rank_skill_docs(docs, "агрегируй таблицу csv", 2, names=names)
    assert idx[0] == 1


# ── 6. фиксы самопроверки: run-scoped создания, memo/prune кэша векторов ───────

def test_created_names_scoped_per_run():
    """Имена созданных навыков не утекают между прогонами (race pop_last_created закрыт)."""
    from src.runtime import run_context as rc

    with rc.request_scope("run_a", "u"):
        sc._note_created("skill_a")
        assert sc.pop_last_created() == "skill_a"      # свой прогон видит своё
    with rc.request_scope("run_a", "u"):
        sc._note_created("leftover")
        with_b_sees = None
    with rc.request_scope("run_b", "u"):
        with_b_sees = sc.pop_last_created()             # чужого имени не видно
    assert with_b_sees == ""
    with rc.request_scope("run_a", "u"):
        assert sc.pop_last_created() == ""              # cleanup вычистил по выходе из scope


class _CountingEmb:
    enabled = True

    def __init__(self):
        self.calls = 0

    def embed(self, text: str):
        self.calls += 1
        return [1.0, 0.0]


def test_emb_cache_memo_avoids_reembed_and_prunes_dead(skl, monkeypatch):
    monkeypatch.setattr(sc, "_emb_memo", {"mtime": None, "data": {}})
    _mk_skill(skl, "memo_skill", _TOOLCODE, md="память кэша")
    sc.sync_registry()  # memo_skill в реестре → «живой» для прюна
    # мёртвый ключ в дисковом кэше — должен уйти при первой же записи
    (skl / ".emb_cache.json").write_text(
        '{"dead_skill": {"mtime": 1.0, "vec": [0.1, 0.2]}}', encoding="utf-8")
    emb = _CountingEmb()
    sc._cached_skill_vecs(["memo_skill"], ["память кэша"], emb)
    assert emb.calls == 1
    sc._cached_skill_vecs(["memo_skill"], ["память кэша"], emb)
    assert emb.calls == 1  # второй запрос — из кэша, без повторного embed
    import json as _json

    disk = _json.loads((skl / ".emb_cache.json").read_text(encoding="utf-8"))
    assert "memo_skill" in disk and "dead_skill" not in disk


def test_distill_verdict_schema_importable():
    from src.llm.structured_outputs import DistillVerdict

    v = DistillVerdict(worth=False, capability="", reason="одноразовая задача")
    assert not v.worth
