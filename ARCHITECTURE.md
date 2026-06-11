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
      └─ deliberate / heavy → [clarify_gate?] → router → (create_skill | skill_selector)
                       clarify_gate — при средней неоднозначности: батч уточнений
                       (маркеры/открытые) перед исполнением; ответы в реестр уточнений
                       прогона, переиспользуются decompose/step/synthesize; нет ответа
                       → разумное допущение. Догон в шаге: инструмент ask_user.
                       skill_selector → decompose → skill_injection
                          → step_executor⟲ (исполнение+валидация ПО ПУНКТАМ)
                          → synthesize ─→ validation → reflect → END               (deliberate)
                                       └→ review (heavy: сквозной ревью deep-моделью)
                                            ├─ проблемы → fix-подшаги → step_executor⟲ → synthesize → validation
                                            └─ чисто → validation
 reflect              запись эпизода (trajectory), извлечение фактов(+тэги, рёбра),
                      рефлексия, саммари, prune, трекинг деградации, авто-self-learning
```

`create_skill`-ветка: ReAct создаёт навык → SGR-ревью + smoke-тест → загрузка (L1 self-improvement, skill library).

## Слои

| Слой | Файлы | Суть |
|---|---|---|
| Когниция / мета-контроль | `agent.py` (goal/reflexion/reason/decompose/step/synthesize ноды) | **5 типов мышления** (fast/reason/deliberate/heavy/clarify) + **reflexion-grounding** (оценка «могу ли достоверно ответить сам» → заземление, анти-галлюцинация); целеполагание, декомпозиция, по-пунктовое исполнение |
| Память | `memory/store.py` (SQLite), `embedder.py`, `vector_index.py` (TurboVec), `feedback.py`, **`memory_tools.py`** | эпизоды/факты(+тэги)/выводы/цели/саммари + граф-рёбра; recall с бюджетом; implicit feedback. **Память-как-tool (3 яруса)**: глобальная (`search_memory`), drill-back полной истории (`recall_history`), временная runtime-scratch (`note_to_self`) — агент сам решает, что подтянуть |
| Навыки | `tools/skill_creation.py`, `skills/*`, **`retrieval.py`** | реестр, защита core, автосинк; **ToolSearch** (BM25S-retrieval навыков при росте библиотеки); `web_search` с контекстным инжинирингом (trafilatura→чанки→BM25S→vector-rerank, полную страницу не кормит); `device_control` |
| Способности-инструменты | **`research.py`** · **`compute.py`** · **`media.py`** · **`mcp_client.py`** | дисциплинированный **research** (план под-вопросов→поиск+сниппеты+чтение→ВЕРИФИКАЦИЯ факта→синтез, зависимая цепочка); **вычислительный слой** `python_exec` (точный счёт в песочнице — rlimits/kill); **vision-чтение фигур PDF** `read_pdf_figures` (рендер→vision, гейт по наличию PDF); **data-MCP само-расширение** `try_connect_discovered` (домен→discover→фильтр релевантности→первый ЖИВОЙ remote-MCP; movie/finance/weather подключаются живьём) |
| Самообучение | `improve/` | forward-харвест few-shots (глоб+**пер-юзер**, двухъярусно с baseline); backward: дифф-credit-assignment → per-node gradients → оптимизация промптов; **per-user backward** (`graph_backward_user`: уроки из неудач юзера → его few-shots); **измеримый accept/revert** (прогон ДО/ПОСЛЕ на кейсах) → ParamStore |
| Трейсинг/диагностика | `tracing/` | спаны по нодам (data/traces.db), самодиагностика, ротация |
| Безопасность | `utils_validation.py` (AST-гейт), `utils.py` (песочница-подпроцесс), `hitl.py` (human-in-the-loop), **`improve/safety.py`** | генерируемый код: AST-запреты + smoke в изолированном процессе (rlimits/kill); side-effect тулы — подтверждение, deny by default; **анти-injection в выводах тулов/MCP/поиска** (`sanitize_tool_output`); запреты обучения (не менять архитектуру/промпты, не учиться на взломе) |
| Внешнее | `external/context.py` | контекст A2A/MCP в состоянии (слот + плумбинг) |
| Обслуживание | `maintenance/dep_update.py` | безопасный авто-апдейт зависимостей с health-check и откатом |
| Интерфейсы | `main.py` (REPL), `bot.py` (Telegram), `server.py` (FastAPI) | общий граф + общая память |

## Self-learning как «обучение графа»

- **Forward**: каждый прогон пишет `run_id`+активации нод в трейс и эпизод (исход+confidence).
  Принятые обдуманные прогоны → few-shots (генерализация без LLM). **Векторизация под
  пользователя**: few-shots пишутся И в персональный стор (`data/user_fewshots.json`,
  ключ = user_id, LRU-cap), И в глобальный. При инъекции в шаг сначала идут ПЕРСОНАЛЬНЫЕ
  примеры (что заходило именно этому человеку), глобальные добивают до k. «Принят» =
  валидирован И не реакция на прошлый плохой ответ (implicit-feedback маркер `[neg]`).
  Промпт-оверрайды остаются глобальными (per-user их крутить — оверфит).
- **Backward** (`improve/graph_learn.py`):
  1. дифференциальная вина: `blame = failRate − successRate` (срабатывающие всегда ноды не виноваты);
  2. `_backward_gradients`: 1 LLM-вызов по батчу → текстовый «градиент» на каждую виноватую ноду;
  3. `optimize_role` для КАЖДОЙ виноватой ноды → проверка плейсхолдеров + LLM-судья + **измеримый ДО/ПОСЛЕ** (прогон на кейсах неудач, сохраняем только при реальном улучшении, иначе откат) → `ParamStore`.
- **Per-user backward** (`graph_backward_user`): из неудач КОНКРЕТНОГО юзера + того, КТО он (профиль/роли), синтезирует корректирующие УРОКИ → его персональные few-shots (ядро заморожено; пишет только в стор этого юзера). Триггер при per-user деградации. Это «оптимизация под пользователя» как метод.
- **Реестр параметров** (`improve/prompt_store.py`, `data/params.json`): prompt-override + few-shots + описания тулов на ноду. Обратимо, revert одной командой, не трогает исходники.
- **Политика оптимизации** (что backward вправе менять):
  - системные промпты КЛЮЧЕВЫХ нод (goal/reflexion/decompose/fast_answer/reason/step_executor/review/clarify_gate) — **ЗАМОРОЖЕНЫ** (это дизайн поведения; `improve.optimize_core_prompts: false`);
  - промпты **саб-агентов-как-тулов** (researcher, …) — оптимизируемы;
  - основной канал улучшения/персонализации — **few-shots** (глобальные + пер-юзер);
  - **архитектура графа никогда не меняется** — структурно: backward пишет только артефакты в ParamStore, не код/граф (судья/анализатор не вправе «выкинуть ноду»);
  - **защита обучения** (`improve/safety.py`): эпизоды-инъекции/джейлбреки исключаются из батча ДО анализа — запрет на «обучение по взлому собственной защиты».
- Батч больше → надёжнее карта вины и богаче few-shots → systematic improvement.

## Конфиг / окружение

- `config.yml`: модели, retries, `memory.*` (recall/embeddings/caps), `skills.protected/autosync`, `improve.*`.
- env: `OPEN_ROUTER_API_KEY` (обяз.), `SEARXNG_URL` (опц., свежий приватный поиск), `OPENAI_API_KEY` (опц., эмбеддинги).

## CLI

- `uvicorn src.server:app` — API (chat/diagnose/memory/traces).
- `python -m src.improve --graph` — backward по графу (credit assignment + per-node оптимизация).
- `python -m src.tracing` — самодиагностика.
- `python -m src.maintenance` — авто-апдейт зависимостей.

## Сделано из прежнего TODO

- **Backward = trace-aware edge-gradient**: tracer пишет выход каждой ноды (`spans.output`), `run_trace(run_id)` даёт цепочку нода→выход; `_format_failure_chains` строит «нода→выход→…→финал», и per-node градиенты раздаются вдоль рёбер (не наивная коактивация).
- **Vision-анализ скриншота**: `device_control.analyze_screen` = `capture_screen` + `media.describe_image` (мультимодальный fast-вызов) одним шагом.
- **MCP/A2A реальный клиент**: `mcp_client` (MultiServerMCPClient) + TRUSTED-allowlist + `discover_mcp` по реестру + human-gate на недоверенные; авто-подключение в `capability_research`.
- **Кроссплатформенность device-ядра**: `open_url/open_app/capture_screen/analyze_screen/notify/speak` имеют бэкенды macOS/Linux/Windows (выбор по `platform.system()`), деградация с подсказкой что доставить.
- **Песочница**: rlimits+kill (всегда) + опциональная syscall-изоляция (bubblewrap/firejail на Linux, sandbox-exec на macOS) — `AGENT_SYSCALL_SANDBOX`.
- **Per-thread chat_history в сервере**: рабочий буфер на `user_id` (поверх долгой памяти).

## Известные границы (TODO)

- Syscall-песочница опциональна и зависит от наличия bwrap/firejail; полноценный gVisor/контейнер на каждый smoke — следующий уровень.
- Работа с УЖЕ ОТКРЫТЫМИ окнами (keystroke/scroll/AX, phone/adb) — пока только macOS; кроссплатформенный UI-automation слой — дальше.
- Оркестрация = выбор 1 из 5 фикс-путей (fast/reason/deliberate/heavy/clarify); свободная динамическая композиция когнитивных модулей — дальше.
- **History-masking** длинного ReAct-контекста (старые наблюдения → заглушки) — отложено: историей сообщений владеет LangGraph `create_agent`, маскинг там = хрупкий хак. Сейчас: сжатие вывода тула (cap) + urllib-first чтение страниц.
- **GraphRAG/LightRAG** для глобальной памяти (level-3 retrieval) — в очереди; сейчас recall recency+relevance+importance + TurboVec-ANN.
