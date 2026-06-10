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

# Грубые цены $/1M токенов (≈ текущий рабочий тир: gemini-2.5-flash-lite /
# deepseek-v4-flash, $0.098–0.10 in / $0.197–0.40 out) — только для оценки.
PRICE_IN = 0.10
PRICE_OUT = 0.30


class TokenTracker(BaseCallbackHandler):
    """Передаётся в config={'callbacks':[tracker]} — ловит usage каждого LLM-вызова."""

    def __init__(self) -> None:
        self.input = 0
        self.output = 0
        self.calls = 0

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        self.calls += 1
        lo = getattr(response, "llm_output", None) or {}
        usage = lo.get("token_usage") or lo.get("usage")
        if usage:
            self.input += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            self.output += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            return
        # фолбэк: usage_metadata на сообщениях
        try:
            for gen in response.generations:
                for g in gen:
                    um = getattr(getattr(g, "message", None), "usage_metadata", None)
                    if um:
                        self.input += um.get("input_tokens", 0)
                        self.output += um.get("output_tokens", 0)
        except Exception:  # noqa: BLE001
            pass

    def snapshot(self) -> tuple[int, int, int]:
        return (self.input, self.output, self.calls)

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
