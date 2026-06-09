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

from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from ..memory import MemoryStore
from ..prompts import OPTIMIZABLE_PROMPTS
from . import prompt_store
from .optimizer import build_optimizer

from ..llm import chat

_cfg = OmegaConf.load("config.yml")
_MODEL = _cfg.get("model", {}).get("name", "gpt-4o-mini")

# Роли, которые умеет оптимизировать пайп: весь реестр обучаемых промптов графа.
_ROLES = OPTIMIZABLE_PROMPTS

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_run_lock = threading.Lock()


class _Verdict(BaseModel):
    is_better: bool = Field(description="Новый промпт строго лучше старого для этих кейсов?")
    reason: str = Field(description="Краткое обоснование")


def _placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(text))


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

        version = prompt_store.save_override(role, proposal, verdict.reason)
        return {"status": "accepted", "role": role, "version": version, "reason": verdict.reason}

    def _failures_batch(self, min_failures: int) -> list[dict]:
        rows = self.store.get_failures(n=20, user_id=None)
        return [
            {"query": r["query"], "answer": r["answer"], "feedback": (r["feedback"] if "feedback" in r.keys() else "")}
            for r in rows
        ]

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
