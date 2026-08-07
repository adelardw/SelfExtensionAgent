"""
Токен-бюджет одного прогона — жёсткий предохранитель от runaway.

Eval поймал случай: deliberate раздробил задачу-знание на 9 веб-ищущих шагов, каждый
шаг — ReAct-цикл, ре-отправляющий растущий контекст → ~1М токенов, $0.11, 18 минут.
Счётчик ШАГОВ этого не ловит (шагов было 9 из 12) — стоимость в ТОКЕНАХ внутри шагов.

Решение: лёгкий callback на всех LLM-клиентах копит токены прогона. ДВА стоп-крана:
  • между нодами — `exhausted(...)` принудительно ведёт к синтезу;
  • ВНУТРИ шага — `step_executor` вооружает (`arm`) бюджет на время исполнения; если шаг сам
    раздул контекст, callback кидает `BudgetExceeded` на следующем LLM-вызове (нужен
    `raise_error=True`, иначе langchain проглотит) → существующий except ведёт к синтезу. Вне
    `arm`/`disarm` callback ничего не прерывает. Сбрасывается в recall_node.

Изоляция по run_id (баг ревью 2a). Состояние прогона (счётчик/таймер) живёт в словаре
`_states[run_id]`, а АКТИВНЫЙ run_id — в contextvar, выставляемом ОДИН раз на границе запроса
(`run_scope` вокруг вызова графа), НЕ внутри ноды. Почему так:
  • contextvar, выставленный во ВНЕШНЕМ контексте (до graph.ainvoke), наследуется всеми нодами и
    callback'ами (вниз по дереву задач — надёжно);
  • `.set()` ВНУТРИ ноды виден только ей, не сестринским нодам (вбок — не наследуется); именно это
    дало −9pp в прошлой попытке (per-task reset «терялся»). Поэтому contextvar здесь только ЧИТАЕМ
    для резолва run_id; мутабельный счётчик — в общем словаре, его записи видны отовсюду.
Без `run_scope` (последовательные REPL/бот/eval) работает общий `_default` — поведение прежнее.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler


class BudgetExceeded(BaseException):
    """Бюджет прогона исчерпан — control-flow сигнал «оборви шаг», а НЕ обычная ошибка.

    Наследует BaseException (как asyncio.CancelledError), чтобы НЕ ловиться широкими `except
    Exception` по пути (research.py/create_agent полны ими — проверено: Exception они глотают,
    BaseException пробивает). Ловит его ТОЛЬКО явный `except runbudget.BudgetExceeded` в
    step_executor → мягкий переход к синтезу."""


class _RunState:
    """Счётчик/таймер/жёсткий обрыв ОДНОГО прогона."""

    __slots__ = ("used", "start", "armed", "hard_tokens", "hard_sec", "lock", "human_wait",
                 "human_since", "human_depth")

    def __init__(self) -> None:
        self.used = 0
        self.start = 0.0
        self.armed = False        # жёсткий обрыв внутри шага (arm/disarm)
        self.hard_tokens = 10**9
        self.hard_sec = 10**9
        self.lock = threading.Lock()
        self.human_wait = 0.0     # ЗАВЕРШЁННЫЕ секунды ожидания человека
        self.human_since = 0.0    # monotonic начала ТЕКУЩЕЙ паузы (0 — паузы нет)
        self.human_depth = 0      # вложенность human_pause (ask внутри HITL)


_registry_lock = threading.Lock()
_states: dict[str, _RunState] = {}
_default = _RunState()            # последовательные вызовы без scope — как раньше (общий)

# run_id берём из ЕДИНОГО run_context (раньше был свой contextvar; теперь общий с clarify/hitl/…
# изоляцией, см. run_context.py). Чистку _states вешаем на выход из request_scope.
from src.runtime import run_context
run_context.register_cleanup(lambda rid: _states.pop(rid, None))


def _state() -> _RunState:
    """Состояние прогона для текущего контекста: по run_id из run_context, иначе общий _default."""
    rid = run_context.current_run_id()
    if rid is None:
        return _default
    st = _states.get(rid)
    if st is None:
        with _registry_lock:
            st = _states.get(rid) or _states.setdefault(rid, _RunState())
    return st


def run_scope(run_id: str):
    """Изолировать бюджет прогона по run_id (алиас к run_context.request_scope для совместимости/
    тестов). На границе сервера/бота используем request_scope(run_id, user_id) — он же изолирует
    clarify/degradation/browser-домены/HITL-гранты."""
    return run_context.request_scope(run_id)


def reset() -> None:
    st = _state()
    with st.lock:
        st.used = 0
        st.start = time.monotonic()
        st.human_wait = 0.0
        st.human_since = 0.0
        st.human_depth = 0


@contextmanager
def human_pause():
    """ЧАСЫ ПРОГОНА СТОЯТ, пока думает ЧЕЛОВЕК (уточнение/HITL-подтверждение).

    Ключевая фишка (как AskUserQuestion в Claude Code): вопрос висит СКОЛЬКО УГОДНО ДОЛГО.
    Без этой паузы wall-clock прогона тикал во время раздумий → пользователь отвечал через
    10 минут, а прогон уже был «исчерпан» и обрывался — вопрос терял смысл. Время ожидания
    человека НЕ его вина и НЕ работа агента: вычитаем его из elapsed/exhausted/hard-abort.
    Реентерабельно (вложенные ask внутри HITL). ВАЖНО: пауза учитывается НА ЛЕТУ (human_since),
    а не только по выходу — иначе наблюдатель (дедлайн тула) во время раздумий видит нули и
    обрывает ожидание (поймано тестом: «человек думает 2.5с» падало на дедлайне 1с)."""
    st = _state()
    with st.lock:
        if st.human_depth == 0:
            st.human_since = time.monotonic()
        st.human_depth += 1
    try:
        yield
    finally:
        with st.lock:
            st.human_depth -= 1
            if st.human_depth <= 0:
                st.human_depth = 0
                if st.human_since:
                    st.human_wait += time.monotonic() - st.human_since
                st.human_since = 0.0


def add(n: int) -> None:
    if n <= 0:
        return
    st = _state()
    with st.lock:
        st.used += n


def used() -> int:
    return _state().used


def over(limit: int) -> bool:
    return _state().used >= limit


def elapsed() -> float:
    """Секунд РАБОТЫ с начала прогона (reset), БЕЗ времени ожидания человека (human_pause).
    Для wall-clock дедлайна против медленного heavy — но не против думающего пользователя."""
    st = _state()
    return max(0.0, time.monotonic() - st.start - _human_total(st)) if st.start else 0.0


def _human_total(st) -> float:
    """Завершённые паузы + ТЕКУЩАЯ (если идёт) — «на лету», для наблюдателей-дедлайнов."""
    live = (time.monotonic() - st.human_since) if st.human_depth and st.human_since else 0.0
    return st.human_wait + live


def human_wait_seconds() -> float:
    """Сколько секунд прогон простоял в ожидании ответа человека (для трейса/диагностики)."""
    return _human_total(_state())


def exhausted(token_limit: int, sec_limit: float) -> bool:
    """Бюджет исчерпан по токенам ИЛИ по времени — оба стоп-крана разом."""
    st = _state()
    # human_wait вычитаем: ожидание ответа человека не «съедает» бюджет прогона (вопрос висит
    # сколько угодно — как AskUserQuestion в Claude Code).
    return st.used >= token_limit or bool(
        st.start and (time.monotonic() - st.start - _human_total(st)) >= sec_limit)


def arm(token_limit: int, sec_limit: float) -> None:
    """Включить жёсткий обрыв на время исполнения шага (с лимитами)."""
    st = _state()
    st.hard_tokens, st.hard_sec = token_limit, sec_limit
    st.armed = True


def disarm() -> None:
    _state().armed = False


class BudgetCallback(BaseCallbackHandler):
    """Копит in+out токены прогона. Вешается на каждый chat-клиент (см. llm.chat).
    Резолвит прогон через contextvar → корректно относит токены к своему run_id под сервером.

    raise_error=True ОБЯЗАТЕЛЕН: langchain по умолчанию (raise_error=False) ЛОВИТ и проглатывает
    исключения из callback'а — без него BudgetExceeded не прервёт LLM-вызов (см. память
    reference-langchain-callback-abort). Срабатывает только когда шаг ВООРУЖЁН (arm) и бюджет
    прогона уже исчерпан, поэтому в норме (disarm) ничего не кидает."""

    raise_error = True

    def _maybe_abort(self) -> None:
        # Стреляет в on_*_start, т.е. ПЕРЕД следующим LLM-вызовом (между раундами шага), а не
        # мид-колл — in-flight вызов прервать нельзя. Для runaway (много раундов с растущим
        # контекстом) это и нужно; одиночный монструозный вызов держит wait_for(deadline).
        st = _state()
        if st.armed and (st.used >= st.hard_tokens or
                         (st.start and (time.monotonic() - st.start - _human_total(st)) >= st.hard_sec)):
            raise BudgetExceeded(f"бюджет прогона исчерпан ({st.used} ток / {elapsed():.0f}с)")

    # ChatOpenAI зовёт on_chat_model_start (НЕ on_llm_start!) — перехватываем оба.
    def on_chat_model_start(self, *args, **kwargs) -> None:  # noqa: ANN002
        self._maybe_abort()

    def on_llm_start(self, *args, **kwargs) -> None:  # noqa: ANN002
        self._maybe_abort()

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        # ВЕСЬ подсчёт токенов — defensive: raise_error=True делает фатальной любую ошибку
        # callback'а, а кривой usage от нестандартного провайдера/прокси (usage не dict) НЕ должен
        # ронять прогон — это лишь телеметрия. Пробрасывается ТОЛЬКО BudgetExceeded из _maybe_abort
        # (on_*_start), не отсюда.
        try:
            lo = getattr(response, "llm_output", None) or {}
            usage = lo.get("token_usage") or lo.get("usage")
            if isinstance(usage, dict):
                add((usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0))
                    + (usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)))
                return
            for gen in response.generations:  # фолбэк: usage_metadata на сообщениях
                for g in gen:
                    um = getattr(getattr(g, "message", None), "usage_metadata", None)
                    if isinstance(um, dict):
                        add(um.get("input_tokens", 0) + um.get("output_tokens", 0))
        except Exception:  # noqa: BLE001 — телеметрия не критична, не валим прогон
            pass


_CB = BudgetCallback()


def callback() -> BudgetCallback:
    """Единый инстанс callback для привязки к LLM-клиентам."""
    return _CB
