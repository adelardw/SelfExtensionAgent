# Архитектура self-extension-agent

Самораширяющийся, самообучающийся агент на LangGraph. Граф агента трактуется как
**обучаемая программа**: прогон = forward pass (трейс активаций), а self-learning —
backward pass по этому трейсу.

## Forward-граф (один запрос)

```
START
 └─ recall            память (эпизоды/факты/выводы/цели/саммари) + implicit feedback + external ctx
 └─ goal              целеполагание: aim + «стоящая» цель + rubric (держится в контексте)
 └─ reflexion         Self-Reflexion Choice: выбор ТИПА мышления по анализу задачи
      ├─ fast      → fast_answer ───────────────────────────────→ reflect → END   (System 1, дёшево)
      ├─ clarify   → fast_answer ───────────────────────────────→ reflect → END   (переспросить)
      ├─ reason    → reason ─────────────────→ validation ──────→ reflect → END   (System 2, без тулов)
      └─ deliberate→ router → (create_skill | skill_selector)
                       skill_selector → decompose → skill_injection
                          → step_executor⟲ (исполнение+валидация ПО ПУНКТАМ)
                          → synthesize → validation → reflect → END                (инструменты)
 reflect              запись эпизода (trajectory), извлечение фактов(+тэги, рёбра),
                      рефлексия, саммари, prune, трекинг деградации, авто-self-learning
```

`create_skill`-ветка: ReAct создаёт навык → SGR-ревью + smoke-тест → загрузка (L1 self-improvement, skill library).

## Слои

| Слой | Файлы | Суть |
|---|---|---|
| Когниция / мета-контроль | `agent.py` (goal/reflexion/reason/decompose/step/synthesize ноды) | 4 типа мышления (fast/reason/deliberate/clarify), целеполагание, декомпозиция, по-пунктовое исполнение |
| Память | `memory/store.py` (SQLite), `embedder.py`, `vector_index.py` (TurboVec), `feedback.py` | эпизоды/факты(+тэги)/выводы/цели/саммари + граф-рёбра; recall recency+relevance+importance с бюджетом; implicit feedback |
| Навыки | `tools/skill_creation.py`, `skills/*` | реестр, защита core-навыков, автосинк, динамическая загрузка; `web_search`(SearXNG→cloakbrowser), `device_control` |
| Самообучение | `improve/` | forward-харвест few-shots; backward: дифф-credit-assignment по трейсу → per-node textual gradients → multi-node оптимизация промптов (TextGrad/Reflexion) → валидация → ParamStore |
| Трейсинг/диагностика | `tracing/` | спаны по нодам (data/traces.db), самодиагностика, ротация |
| Внешнее | `external/context.py` | контекст A2A/MCP в состоянии (слот + плумбинг) |
| Обслуживание | `maintenance/dep_update.py` | безопасный авто-апдейт зависимостей с health-check и откатом |
| Интерфейсы | `main.py` (REPL), `bot.py` (Telegram), `server.py` (FastAPI) | общий граф + общая память |

## Self-learning как «обучение графа»

- **Forward**: каждый прогон пишет `run_id`+активации нод в трейс и эпизод (исход+confidence).
  Успешные обдуманные прогоны → few-shots (генерализация без LLM).
- **Backward** (`improve/graph_learn.py`):
  1. дифференциальная вина: `blame = failRate − successRate` (срабатывающие всегда ноды не виноваты);
  2. `_backward_gradients`: 1 LLM-вызов по батчу → текстовый «градиент» на каждую виноватую ноду;
  3. `optimize_role` для КАЖДОЙ виноватой ноды её градиентом → проверка плейсхолдеров + LLM-судья → `ParamStore`.
- **Реестр параметров** (`improve/prompt_store.py`, `data/params.json`): prompt-override + few-shots + описания тулов на ноду. Обратимо, версионируется, не трогает исходники.
- Обучаемые ноды: goal, reflexion, decompose, fast_answer, reason, step_executor.
- Батч больше → надёжнее карта вины и богаче few-shots → systematic improvement.

## Конфиг / окружение

- `config.yml`: модели, retries, `memory.*` (recall/embeddings/caps), `skills.protected/autosync`, `improve.*`.
- env: `OPEN_ROUTER_API_KEY` (обяз.), `SEARXNG_URL` (опц., свежий приватный поиск), `OPENAI_API_KEY` (опц., эмбеддинги).

## CLI

- `uvicorn src.server:app` — API (chat/diagnose/memory/traces).
- `python -m src.improve --graph` — backward по графу (credit assignment + per-node оптимизация).
- `python -m src.tracing` — самодиагностика.
- `python -m src.maintenance` — авто-апдейт зависимостей.

## Известные границы (TODO)

- Backward = 1 агрегированный вызов по батчу (аппроксимация GEPA); «чистый» textual gradient вдоль рёбер трейса нода→нода — следующий уровень (нужен захват выходов нод в трейс).
- Оркестрация = выбор 1 из 4 фикс-путей; свободная композиция модулей — дальше.
- Vision-анализ скриншота (`capture_screen` даёт PNG) — нужен multimodal-вызов.
- MCP/A2A: слот есть; реальный клиент + авто-подключение по нехватке экспертизы — только через human-gate (security).
- Per-thread chat_history в сервере не хранится (опора на долгую память).
