"""
SelfLearningPipe — evolutionary-контур (L2 self-improvement).

Цикл: собрать слабые трейсы из памяти → предложить улучшенный промпт
(TextGrad / Reflexion) → ВАЛИДИРОВАТЬ улучшение → принять (override) или отклонить.

Валидация двухступенчатая (без неё self-improvement не имеет смысла):
  1. структурная — новый промпт обязан сохранить все плейсхолдеры {name};
  2. качественная — LLM-судья сравнивает старый и новый промпт на тех же кейсах.
Принимаем, только если обе ступени пройдены. Откат — через prompt_store.revert.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from ..memory import MemoryStore
from src.llm.prompts import OPTIMIZABLE_PROMPTS
from . import prompt_store
from .optimizer import build_optimizer
from .safety import filter_learnable

from src.llm.llm import chat

_cfg = OmegaConf.load("config.yml")
_MODEL = _cfg.get("model", {}).get("name", "gpt-4o-mini")

# Роли, по которым пайп ВИДИТ дефолты (для анализа/градиентов) — весь реестр.
_ROLES = OPTIMIZABLE_PROMPTS

# ПОЛИТИКА ОПТИМИЗАЦИИ (что backward вправе ПЕРЕЗАПИСЫВАТЬ):
#   • системные промпты КЛЮЧЕВЫХ когнитивных нод — ЗАМОРОЖЕНЫ (оставляем как есть:
#     это «дизайн» поведения агента, его не переписываем по метрике послушности);
#   • промпты САБ-АГЕНТОВ-ТУЛОВ (researcher и пр.) — оптимизируемы;
#   • основной канал улучшения/персонализации — FEW-SHOTS (глобальные и пер-юзер).
# Архитектуру backward не трогает СТРУКТУРНО: он пишет только артефакты в ParamStore
# (промпты тулов / few-shots), а не код или граф.
TUNABLE_PROMPT_ROLES: set[str] = {"researcher"}  # промпты саб-агентов-как-тулов


def _prompt_tunable(role: str) -> bool:
    """Можно ли ПЕРЕЗАПИСАТЬ системный промпт этой роли (ключевые ноды — нет)."""
    if role in TUNABLE_PROMPT_ROLES:
        return True
    return bool(_cfg.get("improve", {}).get("optimize_core_prompts", False))

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_run_lock = threading.Lock()


class _Verdict(BaseModel):
    is_better: bool = Field(description="Новый промпт строго лучше старого для этих кейсов?")
    reason: str = Field(description="Краткое обоснование")


class _ABVerdict(BaseModel):
    better: Literal["new", "old", "tie"] = Field(description="Какой ответ лучше: 'new'=B, 'old'=A, 'tie'=равно")
    reason: str = Field(description="Кратко почему")


def _placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(text))


def _fill_placeholders(text: str) -> str:
    """Нейтрально заполняет {плейсхолдеры} для before/after-теста: обе версии получают
    ОДИНАКОВЫЙ контекст, поэтому разница в ответах объясняется ТОЛЬКО изменением промпта."""
    return _PLACEHOLDER_RE.sub("(контекст опущен для теста)", text)


class SelfLearningPipe:
    def __init__(self, store: MemoryStore):
        self.store = store
        self._key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._judge = None  # ленивая инициализация: не строим клиент без ключа
        self.optimizer = None

    def _ensure_clients(self) -> None:
        if self.optimizer is None:
            self.optimizer = build_optimizer(_cfg.get("improve", {}).get("optimizer", "textgrad"))
        if self._judge is None:
            self._judge = chat("fast", 0).with_structured_output(_Verdict)

    def _judge_improvement(self, role: str, old: str, new: str, failures: list[dict]) -> _Verdict:
        cases = "\n".join(
            f"- Запрос: {f['query'][:200]} | замечание: {f.get('feedback', '') or '(нет)'}"
            for f in failures
        )
        prompt = (
            f"Сравни два системных промпта агента '{role}'. Кейсы, где старый давал слабый "
            f"результат:\n{cases}\n\n=== СТАРЫЙ ===\n{old}\n\n=== НОВЫЙ ===\n{new}\n\n"
            f"Новый промпт строго лучше для устранения этих неудач и при этом остаётся общим "
            f"и не теряет важных инструкций? Будь строг: при сомнении — не лучше."
        )
        return self._judge.invoke(prompt)

    def _before_after_eval(self, old: str, new: str, failures: list[dict], k: int = 3) -> dict:
        """
        ИЗМЕРИМЫЙ гейт (ход юзера: внутренний сценарный тест ДО/ПОСЛЕ, не мнение о тексте).
        На кейсах неудач генерим ответ под СТАРЫМ и НОВЫМ промптом, судья выбирает лучший.
        Принимаем только при ЧИСТОМ улучшении (after > before) — иначе откат (не сохраняем).
        Дёшево и ограниченно: ≤k кейсов, фиксированная модель fast.
        """
        gen = chat("fast", 0)
        judge = chat("fast", 0).with_structured_output(_ABVerdict)
        old_sys, new_sys = _fill_placeholders(old), _fill_placeholders(new)
        before = after = 0
        cases = failures[:k]
        for f in cases:
            q = (f.get("query") or "")[:300]
            try:
                a_old = gen.invoke([SystemMessage(content=old_sys), HumanMessage(content=q)]).content
                a_new = gen.invoke([SystemMessage(content=new_sys), HumanMessage(content=q)]).content
                v = judge.invoke(
                    f"Запрос: {q}\nЗамечание к прошлому ответу: {f.get('feedback', '') or '(нет)'}\n\n"
                    f"Ответ A (старый промпт):\n{str(a_old)[:800]}\n\n"
                    f"Ответ B (новый промпт):\n{str(a_new)[:800]}\n\n"
                    f"Какой ответ ЛУЧШЕ решает запрос и устраняет замечание?"
                )
                if v.better == "new":
                    after += 1
                elif v.better == "old":
                    before += 1
            except Exception:  # noqa: BLE001
                continue  # сбой кейса не валит весь гейт
        return {"before": before, "after": after, "n": len(cases), "improved": after > before}

    def optimize_role(self, role: str, failures: list[dict], gradient: str = "", accept: bool = True) -> dict:
        """
        Оптимизирует промпт ОДНОЙ роли: optimize (с локальным градиентом) → проверка
        плейсхолдеров → LLM-судья → сохранение в ParamStore. Переиспользуется run() и
        graph_backward() (per-node textual gradient).
        """
        if role not in _ROLES:
            return {"status": "error", "role": role, "reason": "unknown role"}
        if not self._key:
            return {"status": "error", "role": role, "reason": "нет API-ключа"}
        # Ключевые ноды агента — системный промпт ЗАМОРОЖЕН (политика): backward их
        # не переписывает. Их канал улучшения — few-shots (forward-харвест).
        if not _prompt_tunable(role):
            return {"status": "frozen", "role": role,
                    "reason": "системный промпт ключевой ноды не оптимизируется (только few-shots)"}
        # Не учимся на попытках взлома собственной защиты.
        failures = filter_learnable(failures)
        if not failures:
            return {"status": "skipped", "role": role, "reason": "после фильтра безопасности кейсов не осталось"}

        self._ensure_clients()
        default = _ROLES[role]
        current = prompt_store.get_prompt(role, default)
        required = _placeholders(default)

        proposal = self.optimizer.optimize(role, current, failures, gradient=gradient)
        if not proposal:
            return {"status": "failed", "role": role, "reason": "оптимизатор пуст"}

        missing = required - _placeholders(proposal)
        if missing:
            return {"status": "rejected", "role": role, "reason": f"потеряны плейсхолдеры {sorted(missing)}"}

        verdict = self._judge_improvement(role, current, proposal, failures)
        if not (accept and verdict.is_better):
            return {"status": "proposed", "role": role, "accepted": False, "reason": verdict.reason}

        # ИЗМЕРИМЫЙ ГЕЙТ (ход юзера #19): не доверяем одному мнению судьи о тексте —
        # прогоняем ДО/ПОСЛЕ на кейсах. Сохраняем ТОЛЬКО при реальном улучшении (иначе откат).
        ab = {"skipped": True}
        if _cfg.get("improve", {}).get("measure_before_after", True):
            ab = self._before_after_eval(current, proposal, failures)
            if not ab["improved"]:
                return {"status": "rejected", "role": role, "accepted": False, "ab": ab,
                        "reason": f"before/after не показал улучшения (после {ab['after']} ≤ до {ab['before']})"}

        version = prompt_store.save_override(role, proposal, verdict.reason)
        return {"status": "accepted", "role": role, "version": version, "reason": verdict.reason, "ab": ab}

    def _failures_batch(self, min_failures: int) -> list[dict]:
        rows = self.store.get_failures(n=20, user_id=None)
        batch = [
            {"query": r["query"], "answer": r["answer"], "feedback": (r["feedback"] if "feedback" in r.keys() else "")}
            for r in rows
        ]
        return filter_learnable(batch)  # без попыток взлома защиты

    def run(self, role: str = "step_execution", min_failures: int = 3, accept: bool = True) -> dict:
        if not self._key:
            return {"status": "error", "reason": "нет OPEN_ROUTER_API_KEY — оптимизация невозможна"}
        failures = self._failures_batch(min_failures)
        if len(failures) < min_failures:
            return {"status": "skipped", "reason": f"мало неудач ({len(failures)}/{min_failures})"}
        return self.optimize_role(role, failures, accept=accept)


def maybe_auto_improve(store: MemoryStore, degrading: bool = False) -> None:
    """
    Фоновый авто-триггер само-улучшения. НЕ запускается каждую итерацию — только когда
    зафиксирована ДЕГРАДАЦИЯ качества (degrading=True). Запуск при НЕАКТИВНОСТИ — через
    отдельный idle-триггер (см. server.py) или вручную `python -m src.improve --graph`.
    Работает в отдельном потоке, не блокируя ответ.
    """
    imp = _cfg.get("improve", {})
    if not imp.get("auto", False):
        return
    if not degrading:
        return  # триггер только по деградации; обычные ходы ничего не запускают
    if store.failure_count() < imp.get("min_failures", 3):
        return
    if not _run_lock.acquire(blocking=False):
        return  # уже идёт оптимизация

    def _worker():
        try:
            from .graph_learn import batch_optimize  # ленивый импорт (циклы)

            res = batch_optimize(store, min_batch=imp.get("min_failures", 3))
            print(f"[SelfLearning] auto graph-backward: {res}")
        finally:
            _run_lock.release()

    threading.Thread(target=_worker, daemon=True).start()


_user_lock = threading.Lock()


def maybe_improve_user(store: MemoryStore, user_id: str, degrading: bool = False) -> None:
    """
    Фоновый PER-USER авто-триггер (оптимизация под конкретного пользователя). Запускается,
    когда у ЭТОГО юзера зафиксирована деградация качества и накопилось достаточно неудач.
    Пишет только персональные few-shots этого юзера (ядро не трогает). Не блокирует ответ.
    """
    imp = _cfg.get("improve", {})
    if not imp.get("auto", False) or not degrading or not user_id:
        return
    min_f = imp.get("min_failures_user", imp.get("min_failures", 3))
    if len(store.get_failures(n=40, user_id=user_id)) < min_f:
        return
    if not _user_lock.acquire(blocking=False):
        return  # уже идёт персональная оптимизация

    def _worker():
        try:
            from .graph_learn import graph_backward_user  # ленивый импорт (циклы)

            res = graph_backward_user(store, user_id, min_batch=min_f)
            print(f"[SelfLearning] per-user backward [{user_id}]: "
                  f"{res.get('status')} {res.get('lessons_stored', [])}")
        finally:
            _user_lock.release()

    threading.Thread(target=_worker, daemon=True).start()
