"""
Единый per-request контекст (run_id + user_id) — изоляция процессно-глобального состояния под
КОНКУРЕНТНЫМ сервером (FastAPI `asyncio.create_task`) и Telegram (aiogram).

Выставляется ОДИН раз на ГРАНИЦЕ запроса (`request_scope` в server.py/bot.py вокруг вызова графа),
наследуется ВНИЗ по asyncio-задачам — НЕ внутри ноды (`.set()` в ноде не виден сёстрам, см.
runbudget). Модули резолвят `current_run_id()`/`current_user_id()` и держат per-request состояние в
словарях по ключу (а не в голых глобалах) → два одновременных клиента не затирают друг другу ledger
уточнений, анти-тайпсквоттинг-домены, счётчики, гранты HITL. clarify/interaction УЖЕ изолированы
своими contextvar — здесь не дублируются.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Optional

_run_id: ContextVar[Optional[str]] = ContextVar("sea_run_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("sea_user_id", default=None)
_cleanups: list[Callable[[str], None]] = []


def current_run_id() -> Optional[str]:
    return _run_id.get()


def current_user_id() -> Optional[str]:
    return _user_id.get()


def register_cleanup(fn: Callable[[str], None]) -> None:
    """Модуль регистрирует очистку своего per-run состояния (зовётся по выходе из request_scope)."""
    _cleanups.append(fn)


# ── TAINT: в прогон попал НЕДОВЕРЕННЫЙ внешний контент (веб/документ/чужой репо/MCP) ──────────
# Нужен для гейта python_exec: инжектнутый из такого контента вызов на macOS (rlimits-only) мог бы
# читать ФС и слать наружу → требуем HITL-подтверждение (кроме полного auto). Скоуп — по run_id.
_tainted: dict[str, bool] = {}


def mark_external_content() -> None:
    """Пометить текущий прогон: обработан недоверенный внешний контент."""
    _tainted[current_run_id() or "_default"] = True


def external_content_seen() -> bool:
    """Был ли в прогоне недоверенный внешний контент (веб/док/репо/MCP)."""
    return _tainted.get(current_run_id() or "_default", False)


register_cleanup(lambda rid: _tainted.pop(rid, None))


# ── INTRA-NODE трейс: события внутри нод, которых НЕ видно в node-/tool-трейсе ──
# Покрывает (1) дисциплинированный поиск agentic_research (план/под-вопрос/чтение/verify/синтез —
# это chains/urllib, не «тулы») и (2) раунды step_executor (_exec_direct/_exec_compose: где реально
# уходят секунды шага — генерация vs тул-раунды vs ретраи). Эмитим сюда → сервер кладёт в трейс UI,
# видно где время. Скоуп по run_id (как taint), наследуется вниз по asyncio.
_research: dict[str, list] = {}


def research_emit(text: str) -> None:
    """Событие research-цикла для трейса. Кап, чтобы лог не рос бесконечно при многих под-вопросах."""
    log = _research.setdefault(current_run_id() or "_default", [])
    log.append(text)
    if len(log) > 80:
        del log[: len(log) - 80]


def research_log() -> list:
    return list(_research.get(current_run_id() or "_default", []))


register_cleanup(lambda rid: _research.pop(rid, None))


# ── СЧЁТЧИК TOOL-ВЫЗОВОВ прогона (Hermes-порт: триггер ретроспективной дистилляции) ──
# step_executor нотирует реально сделанные вызовы инструментов; reflect по порогу (5+)
# решает, стоила ли задача выделения переиспользуемого навыка из траектории. Скоуп по run_id.
_tool_calls: dict[str, int] = {}


def note_tool_calls(n: int = 1) -> None:
    """Зафиксировать n тул-вызовов текущего прогона."""
    key = current_run_id() or "_default"
    _tool_calls[key] = _tool_calls.get(key, 0) + max(0, int(n))


def tool_calls_count() -> int:
    """Сколько тул-вызовов сделано в текущем прогоне."""
    return _tool_calls.get(current_run_id() or "_default", 0)


# ИМЕНА вызванных инструментов прогона: заземление «что агент реально делал». Без них
# внешний наблюдатель (тестовый стенд/судья) видит только текст ответа и вынужден ГАДАТЬ
# («считал ли python_exec или в уме?») — судьи так и писали «used_compute_tool: неясно».
_tool_names: dict[str, list[str]] = {}


def note_tool_names(names: list[str]) -> None:
    """Зафиксировать имена вызванных инструментов (из ЛЮБОЙ ноды: act/step/research)."""
    if not names:
        return
    lst = _tool_names.setdefault(current_run_id() or "_default", [])
    lst.extend(str(n) for n in names if n)
    del lst[:-60]  # кап на длинный прогон


def tool_names() -> list[str]:
    """Имена инструментов, вызванных в текущем прогоне (в порядке вызова)."""
    return list(_tool_names.get(current_run_id() or "_default", []))


register_cleanup(lambda rid: _tool_names.pop(rid, None))


register_cleanup(lambda rid: _tool_calls.pop(rid, None))


# ── СТАТУС ПОИСКА прогона (гейт актуальности, из мульти-агентной валидации) ─────
# web_search нотирует исход каждого вызова. Синтез читает статистику: время-чувствительный
# запрос при «поиск дёргали, но 0 успешных» ОБЯЗАН выйти с hedge («данные из памяти, могут
# быть устаревшими»), а не подавать числа как текущие. Вскрыто судьёй: агент писал «поиск
# подтвердил» при мёртвом поиске и выдавал ставку 2024 года как «на сегодня». Скоуп по run_id.
_search_calls: dict[str, list[int]] = {}   # run_id -> [attempts, successes]


def note_search_attempt() -> None:
    """Попытка поиска — считается НА ВХОДЕ в вызов (р.5: research отменял зависший поиск по
    wait_for ДО пост-нотации → попытки не считались → circuit-breaker не размыкался, 8 поисков
    подряд об мёртвые бэкенды)."""
    _search_calls.setdefault(current_run_id() or "_default", [0, 0])[0] += 1


def note_search_success() -> None:
    """Успешный исход поиска (результаты получены)."""
    _search_calls.setdefault(current_run_id() or "_default", [0, 0])[1] += 1


def note_search_result(ok: bool) -> None:
    """Совместимость: попытка + исход одним вызовом."""
    note_search_attempt()
    if ok:
        note_search_success()


def search_stats() -> tuple[int, int]:
    """(попыток, успешных) поисковых вызовов текущего прогона."""
    st = _search_calls.get(current_run_id() or "_default", [0, 0])
    return st[0], st[1]


register_cleanup(lambda rid: _search_calls.pop(rid, None))


# ── СТАТУС ЧТЕНИЯ СТРАНИЦ/ДОКУМЕНТОВ прогона (анти-фабрикация, валидация р.3) ──
# browse/read_url нотируют успешные чтения. Если запрос требует данных ИЗ ДОКУМЕНТА
# (is_doc_extraction), а успешных чтений в прогоне НОЛЬ — синтез обязан честно отказаться
# от «точных чисел из статьи», а не фабриковать их из памяти с ложной атрибуцией к таблицам
# (живой провал: выдуманные значения «из Table 2/3» Self-RAG). Скоуп по run_id.
_page_reads: dict[str, list[int]] = {}   # run_id -> [attempts, successes]
_read_urls: dict[str, list[str]] = {}    # run_id -> УСПЕШНО прочитанные URL (сверка named-источника)


def note_page_read(ok: bool, url: str = "") -> None:
    """Зафиксировать исход одного чтения страницы/документа текущего прогона (+URL успеха:
    р.4 вскрыл дыру «прочитал что-то ≠ прочитал НАЗВАННЫЙ источник» — фабрикация проходила,
    пока гейт смотрел только на счётчик)."""
    key = current_run_id() or "_default"
    st = _page_reads.setdefault(key, [0, 0])
    st[0] += 1
    if ok:
        st[1] += 1
        if url:
            lst = _read_urls.setdefault(key, [])
            lst.append(url)
            del lst[:-40]  # кап (длинные прогоны)


def page_read_stats() -> tuple[int, int]:
    """(попыток, успешных) чтений страниц/документов текущего прогона."""
    st = _page_reads.get(current_run_id() or "_default", [0, 0])
    return st[0], st[1]


def read_urls() -> list[str]:
    """URL успешно прочитанных страниц текущего прогона."""
    return list(_read_urls.get(current_run_id() or "_default", []))


register_cleanup(lambda rid: _page_reads.pop(rid, None))
register_cleanup(lambda rid: _read_urls.pop(rid, None))


# ── АРТЕФАКТЫ: файлы, произведённые агентом для ОТДАЧИ пользователю (xlsx/csv/…) ──
# Агент пишет файл нативным тулом (export_table) → регистрирует МЕТА сюда (id/имя/путь). Сервер
# после стрима копирует список в персистентный run-словарь (как research_log) → GUI рисует кнопку
# «скачать» (GET /artifact/{id}); CLI/TUI печатает сохранённый путь. Сами ФАЙЛЫ переживают cleanup
# (лежат в artifacts/<id>/), чистится лишь реестр контекста. Скоуп по run_id.
_artifacts: dict[str, list] = {}


def artifact_emit(meta: dict) -> None:
    """Зарегистрировать произведённый файл (meta: id/name/path/...) для отдачи пользователю."""
    lst = _artifacts.setdefault(current_run_id() or "_default", [])
    # дедуп по id: повторный экспорт того же файла (доп. лист) обновляет запись, не дублирует
    lst[:] = [m for m in lst if m.get("id") != meta.get("id")]
    lst.append(meta)


def artifacts() -> list:
    return list(_artifacts.get(current_run_id() or "_default", []))


register_cleanup(lambda rid: _artifacts.pop(rid, None))


@contextmanager
def request_scope(run_id: str, user_id: str = ""):
    """Изолировать запрос по run_id (+ user_id). Оборачивать вызов графа на границе сервера/бота."""
    t1 = _run_id.set(run_id)
    t2 = _user_id.set(user_id or "")
    try:
        yield
    finally:
        _run_id.reset(t1)
        _user_id.reset(t2)
        for fn in _cleanups:
            try:
                fn(run_id)
            except Exception:  # noqa: BLE001
                pass
