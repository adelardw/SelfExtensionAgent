# GAIA evaluation — raw results & proofs

Held-out runs on **GAIA** (the general-assistant benchmark, levels 1–3) used in the project's
benchmark numbers. These are **raw per-task logs**, not summaries — open them and check every task.

## Files (proofs)

| File | Tier | Models (`config.yml`) | Date |
|---|---|---|---|
| `gaia100_cheap_tier.jsonl` | cheap | fast `google/gemini-2.5-flash-lite`, code `deepseek/deepseek-v4-flash` | 2026-06-14 |
| `gaia100_strong_tier.jsonl` | strong | fast `google/gemini-3.1-flash-lite`, code `z-ai/glm-5.1`, deep `deepseek/deepseek-v4-pro` | 2026-06-14 |

Each line = one task: `{idx, level, mode, ok, gold, final, cost}` — `gold` is the GAIA reference
answer, `final` is the agent's answer, `ok` is the exact-match verdict, `mode` is the thinking mode
the agent chose, `cost` is the `usage.py` flat-rate estimate. 100 tasks each, **0 errored rows**.

## Results (n=100 each)

| Tier | Overall (95% Wilson) | L1 (n=37) | L2 (n=37) | L3 (n=26) | Cost |
|---|---|---|---|---|---|
| cheap | 20.0% [13.3–28.9%] | 41% | 5% | 12% | ~$0.81 (realistic) |
| **strong** | **33.0% [24.6–42.7%]** | **49%** | **38%** | 4% | ≈$2.8 (real-price estimate)\* |

The strong tier is **+13pp overall**, driven by **L2 5%→38%** (multi-hop). Aggregate Wilson
intervals overlap → the gain is real but not yet significant at n=100; the L2 gap is solid.

\* Cost note: per-call tokens were **not** logged per role, only a flat-rate `cost` per task. The
cheap tier's $0.81 matches real prices (~$0.10/$0.40 + $0.09/$0.18). The strong tier's flat log
reads $0.45, but at real prices (gemini-3.1-flash-lite $0.25/$1.50, glm-5.1 $0.98/$3.08) the run's
~4.1M input / 0.12M output tokens cost **≈$1.2–4.4 (~$2.8)**; the exact figure is the OpenRouter bill.

## How the runs were produced

```bash
# fault-tolerant runner: survives a native crash, resumes from the JSONL (offset = lines already done)
AGENT_EVAL_MODE=1 AGENT_NO_BROWSER=1 \
  python scripts/gaia_resilient.py 100 --jsonl data/eval/gaia100.jsonl
```

- `EVAL_MODE` — deterministic eval path; `NO_BROWSER` — no physical browser (headless fetch only).
- **Default budget** (120k tokens / 150s / 8 steps per task) — not inflated for the benchmark.
- The tier is whatever `config.yml` points to at run time (swap models there → re-run).

## Verify the numbers yourself (no network)

```bash
python scripts/gaia_summary.py eval_results/gaia/gaia100_cheap_tier.jsonl \
                               eval_results/gaia/gaia100_strong_tier.jsonl
```

Prints accuracy + Wilson 95% CI + per-level breakdown straight from the JSONL above.

---

# GAIA — сырые результаты и пруфы (Русская версия)

Held-out прогоны на **GAIA** (бенчмарк общего ассистента, уровни 1–3), на которых построены
числа в README. Это **сырые логи по каждой задаче**, не сводка — откройте и проверьте каждую.

## Файлы (пруфы)

| Файл | Тир | Модели (`config.yml`) | Дата |
|---|---|---|---|
| `gaia100_cheap_tier.jsonl` | дешёвый | fast `gemini-2.5-flash-lite`, code `deepseek-v4-flash` | 2026-06-14 |
| `gaia100_strong_tier.jsonl` | сильный | fast `gemini-3.1-flash-lite`, code `glm-5.1`, deep `deepseek-v4-pro` | 2026-06-14 |

Строка = задача: `{idx, level, mode, ok, gold, final, cost}` — `gold` эталонный ответ GAIA,
`final` ответ агента, `ok` вердикт точного совпадения, `mode` выбранный тип мышления,
`cost` flat-оценка `usage.py`. По 100 задач, **0 строк с ошибками**.

## Результаты (n=100 каждый)

| Тир | Overall (95% Wilson) | L1 (n=37) | L2 (n=37) | L3 (n=26) | Стоимость |
|---|---|---|---|---|---|
| дешёвый | 20.0% [13.3–28.9%] | 41% | 5% | 12% | ~$0.81 (реалистично) |
| **сильный** | **33.0% [24.6–42.7%]** | **49%** | **38%** | 4% | ≈$2.8 (оценка по реальным ценам)\* |

Сильный тир — **+13pp overall**, главным образом за счёт **L2 5%→38%** (multi-hop). Агрегатные
Wilson-интервалы пересекаются → прирост реален, но на n=100 ещё не статзначим; по L2 разрыв уверенный.

\* По стоимости: per-call токены по ролям **не** логировались, только flat-`cost` на задачу.
$0.81 дешёвого тира совпадает с реальными ценами. У сильного flat-лог даёт $0.45, но по реальным
ценам (gemini-3.1-flash-lite $0.25/$1.50, glm-5.1 $0.98/$3.08) на ~4.1M вход / 0.12M выход токенов
выходит **≈$1.2–4.4 (~$2.8)**; точная сумма — в биллинге OpenRouter.

## Как получены прогоны

```bash
AGENT_EVAL_MODE=1 AGENT_NO_BROWSER=1 \
  python scripts/gaia_resilient.py 100 --jsonl data/eval/gaia100.jsonl
```

Отказоустойчивый раннер: переживает нативный краш, резюмируется по JSONL (offset = уже сделанные
строки). `EVAL_MODE` — детерминированный путь eval; `NO_BROWSER` — без физ-браузера; **дефолтный
бюджет** (120k / 150с / 8 шагов), не раздут под бенч. Тир = то, что в `config.yml` на момент прогона.

## Проверить числа самому (без сети)

```bash
python scripts/gaia_summary.py eval_results/gaia/gaia100_cheap_tier.jsonl \
                               eval_results/gaia/gaia100_strong_tier.jsonl
```
