"""L4: проектный ярус навыков (.sea/skills/). Проектный навык грузится/выбирается; глобальные
(src/skills) не ломаются; запись остаётся глобальной (проектные не протекают). Offline."""
import json

import pytest

import src.tools.skill_creation as sc


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    d = tmp_path / ".sea" / "skills" / "projhello"
    d.mkdir(parents=True)
    (d / "projhello.py").write_text(
        "from langchain_core.tools import tool\n\n@tool\ndef proj_hi(name: str) -> str:\n"
        "    '''Поздороваться по имени.'''\n    return f'hi {name}'\n", encoding="utf-8")
    (d / "projhello.md").write_text(
        "# Skill: projhello\nуникумтокензет проектный навык приветствия", encoding="utf-8")
    (tmp_path / ".sea" / "skills" / "registry.json").write_text(
        json.dumps({"projhello": {"description": "проектный", "has_tools": True,
                                  "has_system_prompt": False}}), encoding="utf-8")
    return tmp_path


def test_no_project_dir_is_global_only(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))  # нет .sea/skills
    assert sc._project_skills_dir() is None
    # merged == global (существующие навыки на месте)
    assert "web_search" in sc._merged_registry()
    assert sc._skill_base("web_search") == sc.SKILLS_DIR / "web_search"


def test_project_skill_merged_and_resolved(proj):
    reg = sc._merged_registry()
    assert "projhello" in reg and "web_search" in reg          # проектный + глобальные
    assert sc._skill_base("projhello") == proj / ".sea" / "skills" / "projhello"
    assert sc._skill_base("web_search") == sc.SKILLS_DIR / "web_search"  # глобальный — из src/skills


def test_project_skill_tool_loads(proj):
    tools = sc.get_all_loaded_skill_tools(["projhello"])
    assert any(t.name == "proj_hi" for t in tools)


def test_project_skill_selectable(proj):
    rel = sc.get_relevant_skills_for_prompt("нужен уникумтокензет проектный навык")
    assert "projhello" in rel


def test_save_registry_stays_global(proj):
    """Запись реестра НЕ должна писать проектные навыки в глобальный src/skills/registry.json."""
    glob_before = set(sc._load_registry())          # глобальный (без проектного)
    assert "projhello" not in glob_before           # проектный не в глобальном
    assert "projhello" in sc._merged_registry()      # но виден в merged


# ── L4b/c: создание навыков по скоупу + CRUD-цикл ──────────────────────────────
_TOOLCODE = ("from langchain_core.tools import tool\n\n@tool\ndef pcalc(x: int) -> int:\n"
             "    '''Прибавить 1.'''\n    return x + 1\n")


def test_default_scope_project_in_initialized_project(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    assert sc._default_scope() == "global"           # нет .sea/ → global (аддитивно)
    (tmp_path / ".sea").mkdir()
    assert sc._default_scope() == "project"           # есть .sea/ → project


def test_create_project_skill_full_cycle(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    (tmp_path / ".sea").mkdir()                        # инициализированный проект
    glob_before = set(sc._load_registry())            # снимок глобального реестра

    res = sc.create_skill.invoke({"name": "projcalc", "description": "проектный калькулятор",
                                  "tool_code": _TOOLCODE})
    assert "project" in res
    # лёг в .sea/skills, НЕ в глобальный
    assert (tmp_path / ".sea" / "skills" / "projcalc" / "projcalc.py").exists()
    assert "projcalc" not in sc._load_registry()      # глобальный реестр НЕ тронут
    assert set(sc._load_registry()) == glob_before
    assert "projcalc" in sc._merged_registry()         # но виден в merged
    assert sc._skill_scope("projcalc") == "project"

    # CRUD-цикл для проектного навыка резолвится в проект
    assert "pcalc" in sc.read_skill.invoke({"name": "projcalc"})
    from src.utils import _skill_loadable
    ok, msg = _skill_loadable("projcalc")              # SGR-валидация находит project-навык
    assert ok and "pcalc" in msg
    sc.mark_temporary("projcalc")
    pr = sc._load_reg_at(sc._project_registry_path())
    assert pr["projcalc"]["temporary"] is True         # пометка в ПРОЕКТНОМ реестре
    # удаление — из проекта, глобальный так и не тронут
    sc.delete_skill.invoke({"name": "projcalc"})
    assert not (tmp_path / ".sea" / "skills" / "projcalc").exists()
    assert set(sc._load_registry()) == glob_before
