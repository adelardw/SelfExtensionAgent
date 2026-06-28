"""Синтез доставляет ФАЙЛ, а не обещание (живой баг: «Файл готов к скачиванию» при отсутствии xlsx).

Если шаг экспорта не запустился или упал → артефакта нет. Раньше модель могла выдумать «Файл готов
к скачиванию: …xlsx». Теперь в synthesize_node:
  - если просили файл, артефакта нет, но в собранном есть табличные данные → формируется .xlsx;
  - если данных в таблицу нет → честная пометка вместо ложного обещания файла.

Файловый fallback срабатывает по _wants_file_output && отсутствию артефактов; подменяем
synth_chain.ainvoke (финал) + llm.ainvoke (извлечение CSV) на детерминированные значения.
"""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import src.graph.agent as A
from src.runtime import run_context as RC


class _Resp:
    def __init__(self, content):
        self.content = content


def _state(query, step_results):
    return {"query": query, "step_results": step_results, "memory_context": ""}


def _run_synth(monkeypatch, tmp_path, query, step_results, synth_text, csv_text):
    monkeypatch.chdir(tmp_path)  # artifacts/ пишется в cwd → не сорим в репо
    with RC.request_scope("scope-synthfile", "local"):
        fake_chain = SimpleNamespace(ainvoke=AsyncMock(return_value=_Resp(synth_text)))
        fake_llm = SimpleNamespace(ainvoke=AsyncMock(return_value=_Resp(csv_text)))
        # сигнал «нужен файл» — отдельный эмбеддинг-классификатор (требует ключ/сеть, тестируется
        # своими тестами); здесь его фиксируем True, чтобы проверять именно логику ДОСТАВКИ файла.
        with patch.object(A, "_wants_file_output", return_value=True), \
             patch.object(A, "synth_chain", fake_chain), \
             patch.object(A, "llm", fake_llm):
            out = asyncio.new_event_loop().run_until_complete(
                A.synthesize_node(_state(query, step_results)))
            arts = RC.artifacts()
    return out, arts


def test_file_request_builds_real_xlsx_from_gathered(monkeypatch, tmp_path):
    """Просили файл, артефакта нет, но в собранном есть таблица → создаётся настоящий .xlsx."""
    step_results = [{"goal": "сбор", "result": "Год,Пассажиры\n2014,2000\n2015,2100"}]
    out, arts = _run_synth(monkeypatch, tmp_path,
                           "сделай excel таблицу по метро", step_results,
                           synth_text="Готово, собрал данные за 2014-2015.",
                           csv_text="Год,Пассажиры\n2014,2000\n2015,2100")
    assert len(arts) == 1
    assert arts[0]["name"].endswith(".xlsx")
    assert os.path.exists(arts[0]["path"])


def test_no_tabular_data_no_false_file_claim(monkeypatch, tmp_path):
    """Просили файл, данных в таблицу нет → НЕ плодим пустышку и НЕ врём про готовый файл."""
    step_results = [{"goal": "сбор", "result": "Данные закрыты, цифр не нашёл."}]
    out, arts = _run_synth(monkeypatch, tmp_path,
                           "сделай excel таблицу по метро", step_results,
                           synth_text="Файл готов к скачиванию: metro.xlsx",
                           csv_text="")  # модель не смогла извлечь таблицу
    assert arts == []                                   # пустышку не плодим
    _fa = out["final_answer"]
    assert "не приложен" in _fa and "выдумывать" in _fa  # честно: файла нет + ничего не сочиняем (без отмазок)
