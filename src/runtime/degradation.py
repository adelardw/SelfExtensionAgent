"""
Видимость тихих деградаций (долг ревью #5).

В коде ~540 broad-except, которые ловят сбой и уводят в БЕЗОПАСНЫЙ fallback (reflexion→deliberate,
decompose→один шаг, шаг прерван→синтез, валидация пропущена→принять). Поодиночке это graceful
degradation, но В СУММЕ при системном сбое (кончился ключ эмбеддера, провайдер 5xx) агент не падает,
а МОЛЧА тупеет — и без AGENT_DEBUG это не видно. Это же маскирует другие баги (мёртвый предохранитель
и т.п.).

Здесь — лёгкий ЦЕНТРАЛЬНЫЙ счётчик: каждый критический fallback зовёт note(); общий rate виден через
snapshot()/total() (для /diagnose) и печатается сразу под AGENT_DEBUG. Не меняет поток — только
делает деградацию НАБЛЮДАЕМОЙ.

Счётчик ПРОЦЕССНО-КУМУЛЯТИВНЫЙ (систем-здоровье: «кончился ключ эмбеддера → массовые fallback'и»),
а НЕ по-прогонный: per-run reset под конкурентным сервером затирал бы чужой счётчик (гонка).
Конкурентные note() сериализованы локом → инкременты корректны без гонки.
"""
from __future__ import annotations

import os
import threading
from collections import Counter

_lock = threading.Lock()
_counts: Counter = Counter()


def note(where: str, exc: BaseException | str = "") -> None:
    """Зафиксировать тихую деградацию в точке `where`. exc — исключение/причина (для лога)."""
    with _lock:
        _counts[where] += 1
    if os.getenv("AGENT_DEBUG") == "1":
        detail = type(exc).__name__ if isinstance(exc, BaseException) else str(exc)
        print(f"[Degradation] {where}: {detail}".rstrip(": "))


def snapshot() -> dict:
    """Копия счётчиков {where: count} — для /diagnose и тестов."""
    with _lock:
        return dict(_counts)


def total() -> int:
    with _lock:
        return sum(_counts.values())


def reset() -> None:
    with _lock:
        _counts.clear()
