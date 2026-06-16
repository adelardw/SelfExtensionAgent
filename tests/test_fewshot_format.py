"""
B6: backward-уроки (проза) и forward-примеры (запрос→режим) живут в одном пуле few-shots, но
РЕНДЕРЯТСЯ ПО-РАЗНОМУ — урок не должен попадать в слот «Хороший ответ: <режим>» и сбивать
классификатор маршрутизации.
"""
from src.improve import prompt_store as ps


def test_lessons_render_separately_from_examples(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "USER_FEWSHOTS_FILE", tmp_path / "u.json")
    monkeypatch.setattr(ps, "PARAMS_FILE", tmp_path / "p.json")

    # forward-пример: запрос → mode-метка
    ps.add_user_fewshot("u1", "reflexion", "посчитай факториал 10", "deliberate", 0.8)
    # backward-урок: проза
    ps.add_user_fewshot("u1", "reflexion", "вопросы про погоду",
                        "не отвечай из памяти — нужен веб-поиск свежих данных", 0.9, kind="lesson")

    out = ps.format_fewshots("reflexion", k=6, user_id="u1", query="посчитай")

    # пример отрендерен как запрос→ответ
    assert "Хороший ответ: deliberate" in out
    # урок — в отдельном блоке «Уроки», НЕ как «Хороший ответ: <проза>»
    assert "Уроки" in out
    assert "Хороший ответ: не отвечай из памяти" not in out
    assert "не отвечай из памяти" in out          # урок присутствует, но отдельно


def test_default_kind_is_example(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PARAMS_FILE", tmp_path / "p.json")
    ps.add_fewshot("reflexion", "привет", "fast", 0.7)   # без kind → example
    shots = ps.get_fewshots("reflexion")
    assert shots and shots[0].get("kind") == "example"
