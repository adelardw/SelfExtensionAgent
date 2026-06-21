"""
Учёт расхода токенов LLM через LangChain-callback + персистентная статистика.

TokenTracker накапливает вход/выход по всем вызовам модели за сессию; data/usage.json
хранит all-time. Цены — грубая оценка $ (настраиваемо), т.к. модели разные.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

USAGE_FILE = Path("data/usage.json")

# Грубые цены $/1M токенов — ТОЛЬКО для оценки (единая ставка на все роли).
# Текущий тир: fast=gemini-3.1-flash-lite $0.25/$1.50, code=glm-5.1 $0.98/$3.08,
# deep=deepseek-v4-pro $0.435/$0.87. Базис ниже — по fast (самые частые вызовы);
# реальная стоимость ВЫШЕ оценки, т.к. шаги на glm-5.1 (code) дороже.
PRICE_IN = 0.25
PRICE_OUT = 1.50


class TokenTracker(BaseCallbackHandler):
    """Передаётся в config={'callbacks':[tracker]} — ловит usage каждого LLM-вызова."""

    def __init__(self) -> None:
        self.input = 0
        self.output = 0
        self.calls = 0
        self.started = 0   # сколько LLM-вызовов НАЧАТО (для индикатора «вызов в полёте»: started>calls)
        # K3: видимость авто-prefix-cache. DeepSeek/Gemini/OpenAI-совместимые отдают долю
        # входных токенов, прочитанных из кэша (prompt_tokens_details.cached_tokens). Без
        # этого нельзя измерить эффект стабилизации префикса — диагностика была слепой.
        self.cached = 0

    # on_..._start фиксируют НАЧАЛО вызова → UI видит, что модель сейчас работает, ещё до того,
    # как придут токены (они считаются только на on_llm_end, пачкой по завершении вызова).
    def on_chat_model_start(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.started += 1

    def on_llm_start(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.started += 1

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        self.calls += 1
        lo = getattr(response, "llm_output", None) or {}
        usage = lo.get("token_usage") or lo.get("usage")
        if usage:
            self.input += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            self.output += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            details = usage.get("prompt_tokens_details") or {}
            self.cached += (
                (details.get("cached_tokens", 0) if isinstance(details, dict) else 0)
                or usage.get("cache_read_input_tokens", 0)
            )
            return
        # фолбэк: usage_metadata на сообщениях
        try:
            for gen in response.generations:
                for g in gen:
                    um = getattr(getattr(g, "message", None), "usage_metadata", None)
                    if um:
                        self.input += um.get("input_tokens", 0)
                        self.output += um.get("output_tokens", 0)
                        itd = um.get("input_token_details") or {}
                        if isinstance(itd, dict):
                            self.cached += itd.get("cache_read", 0)
        except Exception:  # noqa: BLE001
            pass

    def snapshot(self) -> tuple[int, int, int]:
        return (self.input, self.output, self.calls)

    @property
    def cache_hit_rate(self) -> float:
        """Доля входных токенов, прочитанных из авто-кэша (0..1). 0 = префикс не кэшируется."""
        return self.cached / self.input if self.input else 0.0

    @property
    def total(self) -> int:
        return self.input + self.output

    def cost(self) -> float:
        return self.input / 1e6 * PRICE_IN + self.output / 1e6 * PRICE_OUT


def cost_of(inp: int, out: int) -> float:
    return inp / 1e6 * PRICE_IN + out / 1e6 * PRICE_OUT


def load_alltime() -> dict:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {"input": 0, "output": 0, "calls": 0}


def add_alltime(inp: int, out: int, calls: int) -> dict:
    d = load_alltime()
    d["input"] = d.get("input", 0) + inp
    d["output"] = d.get("output", 0) + out
    d["calls"] = d.get("calls", 0) + calls
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return d
