"""
Токен-бюджет одного прогона — жёсткий предохранитель от runaway.

Eval поймал случай: deliberate раздробил задачу-знание на 9 веб-ищущих шагов, каждый
шаг — ReAct-цикл, ре-отправляющий растущий контекст → ~1М токенов, $0.11, 18 минут.
Счётчик ШАГОВ этого не ловит (шагов было 9 из 12) — стоимость в ТОКЕНАХ внутри шагов.

Решение: лёгкий callback на всех LLM-клиентах копит токены прогона в module-level
счётчике (НЕ contextvar: значение, выставленное в callback, в другом контексте не
видно основной задаче — проверено). Ноды проверяют бюджет и принудительно идут к
синтезу. Сбрасывается в recall_node. Для одного пользователя (REPL/бот — запросы
последовательны) точно; на сервере при гонке счётчик общий → может остановить чуть
раньше (безопасное направление, не runaway).
"""
from __future__ import annotations

import threading
import time

from langchain_core.callbacks import BaseCallbackHandler

_lock = threading.Lock()
_used = 0
_start = 0.0
# Жёсткий обрыв ВНУТРИ шага: callback кидает BudgetExceeded на след. LLM-вызове, когда
# бюджет исчерпан. Включается только на время исполнения шага (arm/disarm), чтобы
# synthesize/validation потом всё равно собрали ответ из того, что есть.
_armed = False
_hard_tokens = 10**9
_hard_sec = 10**9


class BudgetExceeded(Exception):
    """Бюджет прогона исчерпан посреди шага — мягко завершаем шаг."""


def reset() -> None:
    global _used, _start
    with _lock:
        _used = 0
        _start = time.monotonic()


def add(n: int) -> None:
    global _used
    if n <= 0:
        return
    with _lock:
        _used += n


def used() -> int:
    return _used


def over(limit: int) -> bool:
    return _used >= limit


def elapsed() -> float:
    """Секунд с начала прогона (reset). Для wall-clock дедлайна против медленного heavy."""
    return time.monotonic() - _start if _start else 0.0


def exhausted(token_limit: int, sec_limit: float) -> bool:
    """Бюджет исчерпан по токенам ИЛИ по времени — оба стоп-крана разом."""
    return _used >= token_limit or (_start and elapsed() >= sec_limit)


def arm(token_limit: int, sec_limit: float) -> None:
    """Включить жёсткий обрыв на время исполнения шага (с лимитами)."""
    global _armed, _hard_tokens, _hard_sec
    _hard_tokens, _hard_sec = token_limit, sec_limit
    _armed = True


def disarm() -> None:
    global _armed
    _armed = False


class BudgetCallback(BaseCallbackHandler):
    """Копит in+out токены прогона. Вешается на каждый chat-клиент (см. llm.chat)."""

    def _maybe_abort(self) -> None:
        # Жёсткий обрыв ВНУТРИ шага: если бюджет исчерпан — не делаем след. вызов.
        if _armed and exhausted(_hard_tokens, _hard_sec):
            raise BudgetExceeded(f"бюджет прогона исчерпан ({_used} ток / {elapsed():.0f}с)")

    # ChatOpenAI зовёт on_chat_model_start (НЕ on_llm_start!) — перехватываем оба.
    def on_chat_model_start(self, *args, **kwargs) -> None:  # noqa: ANN002
        self._maybe_abort()

    def on_llm_start(self, *args, **kwargs) -> None:  # noqa: ANN002
        self._maybe_abort()

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        lo = getattr(response, "llm_output", None) or {}
        usage = lo.get("token_usage") or lo.get("usage")
        if usage:
            add((usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0))
                + (usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)))
            return
        try:  # фолбэк: usage_metadata на сообщениях
            for gen in response.generations:
                for g in gen:
                    um = getattr(getattr(g, "message", None), "usage_metadata", None)
                    if um:
                        add(um.get("input_tokens", 0) + um.get("output_tokens", 0))
        except Exception:  # noqa: BLE001
            pass


_CB = BudgetCallback()


def callback() -> BudgetCallback:
    """Единый инстанс callback для привязки к LLM-клиентам."""
    return _CB
