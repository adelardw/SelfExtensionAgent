"""
Оптимизаторы промптов для self-learning пайпа.

Два бэкенда за единым интерфейсом (паттерн как у эмбеддера — graceful):
  • TextGradOptimizer — «текстовые градиенты» (Stanford TextGrad). Точный метод,
    но требует рабочего LLM-движка.
  • ReflexionOptimizer — фолбэк на обычный LLM-рерайт по критике (Reflexion-стиль).
    Работает всегда, пока есть LLM.

Фабрика build_optimizer() выбирает TextGrad, если он импортируется и движок
строится; иначе — Reflexion.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol

from omegaconf import OmegaConf

from ..llm import OPENROUTER_BASE as _OPENROUTER

_cfg = OmegaConf.load("config.yml")
_MODEL = _cfg.get("code_model", {}).get("name", "gpt-4o-mini")


def _api_key() -> Optional[str]:
    return os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def _format_failures(failures: list[dict]) -> str:
    parts = []
    for i, f in enumerate(failures, 1):
        parts.append(
            f"[Кейс {i}] Запрос: {f.get('query', '')[:300]}\n"
            f"Слабый ответ: {f.get('answer', '')[:300]}\n"
            f"Замечание валидатора: {f.get('feedback', '(нет)')[:200]}"
        )
    return "\n\n".join(parts)


class PromptOptimizer(Protocol):
    name: str

    def optimize(self, role: str, current: str, failures: list[dict], gradient: str = "") -> Optional[str]:
        ...


def _gradient_block(gradient: str) -> str:
    return f"\n\nЛокальный градиент этой ноды (backward по графу — учти в первую очередь):\n{gradient}" if gradient else ""


class TextGradOptimizer:
    """Оптимизация промпта через textual gradients."""

    name = "textgrad"

    def __init__(self):
        self._engine = None
        try:
            import textgrad as tg
            from textgrad.engine.openai import ChatOpenAI as TGEngine

            key = _api_key()
            if not key:
                raise RuntimeError("нет API-ключа для движка")
            os.environ.setdefault("OPENAI_API_KEY", key)
            self._tg = tg
            self._engine = TGEngine(model_string=_MODEL, base_url=_OPENROUTER)
        except Exception as e:  # noqa: BLE001
            print(f"[TextGrad] недоступен ({e})")

    @property
    def available(self) -> bool:
        return self._engine is not None

    def optimize(self, role: str, current: str, failures: list[dict], gradient: str = "") -> Optional[str]:
        if not self.available:
            return None
        tg = self._tg
        try:
            tg.set_backward_engine(self._engine, override=True)
            prompt_var = tg.Variable(
                current,
                requires_grad=True,
                role_description=f"system prompt of the {role} agent that must be improved",
            )
            instruction = (
                f"This system prompt drives the '{role}' agent. On the cases below it produced "
                f"weak answers. Improve the prompt so such failures stop, keeping it general "
                f"(do not overfit to these exact cases). CRITICAL: preserve every placeholder "
                f"written as {{name}} exactly as-is.{_gradient_block(gradient)}\n\nFailing cases:\n{_format_failures(failures)}"
            )
            loss_fn = tg.TextLoss(instruction)
            loss = loss_fn(prompt_var)
            loss.backward()
            optimizer = tg.TGD(parameters=[prompt_var])
            optimizer.step()
            return prompt_var.value
        except Exception as e:  # noqa: BLE001
            print(f"[TextGrad] optimize failed: {e}")
            return None


class ReflexionOptimizer:
    """Фолбэк: LLM переписывает промпт по словесной критике неудач."""

    name = "reflexion"

    def __init__(self):
        from langchain_openai.chat_models import ChatOpenAI

        self._llm = ChatOpenAI(api_key=_api_key(), base_url=_OPENROUTER, model=_MODEL, temperature=0.3)

    @property
    def available(self) -> bool:
        return True

    def optimize(self, role: str, current: str, failures: list[dict], gradient: str = "") -> Optional[str]:
        try:
            msg = (
                f"Ты — оптимизатор промптов. Ниже системный промпт агента '{role}' и кейсы, "
                f"где он дал слабый результат. Перепиши промпт так, чтобы устранить причины "
                f"неудач, сохранив его общим (не подгоняй под конкретные кейсы). "
                f"ОБЯЗАТЕЛЬНО сохрани все плейсхолдеры вида {{name}} без изменений. "
                f"Верни ТОЛЬКО новый текст промпта, без пояснений.{_gradient_block(gradient)}\n\n"
                f"=== ТЕКУЩИЙ ПРОМПТ ===\n{current}\n\n"
                f"=== НЕУДАЧНЫЕ КЕЙСЫ ===\n{_format_failures(failures)}"
            )
            resp = self._llm.invoke(msg)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:  # noqa: BLE001
            print(f"[Reflexion] optimize failed: {e}")
            return None


def build_optimizer(prefer: str = "textgrad") -> PromptOptimizer:
    """Фабрика: TextGrad если доступен, иначе Reflexion."""
    if prefer == "textgrad":
        tgo = TextGradOptimizer()
        if tgo.available:
            return tgo
        print("[Optimizer] TextGrad недоступен → Reflexion-фолбэк.")
    return ReflexionOptimizer()
