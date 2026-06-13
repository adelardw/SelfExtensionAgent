# Архитектура self-extension-agent

Самораширяющийся, самообучающийся агент на LangGraph. Граф агента трактуется как
**обучаемая программа**: прогон = forward pass (трейс активаций), а self-learning —
backward pass по этому трейсу.

## Forward-граф (один запрос)

```
START
 └─ recall            память (эпизоды/факты/выводы/цели/саммари) + implicit feedback + external ctx
                      + AutoRAG (БЗ юзера + вложения сессии, BM25 + sanitize) + сброс журнала взаимодействий.
                      УСЛОВНЫЙ («recall не всегда»): персона-факты всегда, ассоциативная память — по
                      гейту релевантности (recall_gate); GraphRAG-lite (densify fact↔fact + spreading-
                      activation). Запрос эмбеддится ОДИН раз → переиспускается в gate/graph И intent-роутере.
 └─ goal              целеполагание: aim + «стоящая» цель + rubric (держится в контексте)
 └─ reflexion         Self-Reflexion Choice: выбор ТИПА мышления по анализу задачи
                      (+ бандит-прайор: Beta/Thompson по похожим эпизодам юзера, видит и неудачи;
                      + universal intent-роутер: embedding-kNN кодбук маршрутов любой язык, регэксп=fallback;
                      heavy НЕ предсказывается — ЗАРАБАТЫВАЕТСЯ рантайм-evidence в route_after_synthesize)
      ├─ fast      → fast_answer ───────────────────────────────→ reflect → END   (System 1, дёшево)
      ├─ clarify   → fast_answer ───────────────────────────────→ reflect → END   (переспросить)
      ├─ act       → act ───────────────────────────────────────→ reflect → END   (System 1 с руками:
      │                ОДНО прямое действие 1–2 тулами (BM25-подбор навыка, HITL сохранён);
      │                ни одного вызова тула / ESCALATE → эскалация в deliberate (→ goal)
      ├─ reason    → reason ─────────────────→ validation ──────→ reflect → END   (System 2, без тулов)
      └─ deliberate / heavy → [clarify_gate?] → router → (create_skill | skill_selector)
                       clarify_gate — при средней неоднозначности: батч уточнений
                       (маркеры/открытые) перед исполнением; ответы в реестр уточнений
                       прогона, переиспользуются decompose/step/synthesize; нет ответа
                       → разумное допущение. Догон в шаге: инструмент ask_user.
                       skill_selector → decompose → skill_injection
                          (амортизация: при РЕЦЕПТЕ похожей успешной задачи селектор БЕЗ
                           LLM-вызова; при sim≥0.7 и decompose БЕЗ LLM — план из рецепта)
                          → step_executor⟲ (исполнение+валидация ПО ПУНКТАМ,
                             валидатор видит РЕАЛЬНО вызванные тулы: текст ≠ действие)
                          → synthesize ─→ validation → reflect → END               (deliberate)
                                       └→ review (heavy: сквозной ревью deep-моделью)
                                            ├─ проблемы → fix-подшаги → step_executor⟲ → synthesize → validation
                                            └─ чисто → validation
 reflect              запись эпизода (trajectory + журнал взаимодействий), harvest сигнала
                      (HITL-отказ/clarify-ответ → факты профиля), компиляция РЕЦЕПТА и
                      win/lose применённого, промоушен в коллективный пул, детекция привычки,
                      извлечение фактов(+тэги, рёбра), рефлексия, саммари, prune,
                      трекинг деградации, авто-self-learning
```

`create_skill`-ветка: ReAct создаёт навык → SGR-ревью + smoke-тест → загрузка (L1 self-improvement, skill library).

## Два уровня маршрутизации (НЕ путать)

Маршрутизаций ДВЕ, на разных уровнях. Intent-роутер НЕ выбирает режим мышления — это делает LLM.

```
                          запрос + qvec (эмбеддинг запроса, посчитан в recall ОДИН раз)
                                              │
        ┌─────────────────────────────────────┴─────────────────────────────────────┐
        │                                                                             │
  УРОВЕНЬ 1 — РЕЖИМ МЫШЛЕНИЯ («как думать»)              УРОВЕНЬ 2 — INTENT-СИГНАЛЫ («что за запрос»)
  кто решает: reflexion-LLM (ReflexionDecision)         кто решает: embedding-kNN кодбук (intent.py) на qvec
  + бандит-прайор + similarity few-shots                сигналы: web_grounding / physical_browser /
  выход: fast | reason | act | deliberate | heavy |              play_media / self_contained
         clarify  (+ ambiguity/grounding-оценки)        НЕ режим — ГЕЙТЫ поведения:
        │                                                  • web_grounding → ПЕРЕБИВАЕТ режим на act
        │                                                    (анти-галлюц. ПОЛ; UNION регэксп∨классификатор)
        │                                                  • physical_browser → внутри act: физ-браузер vs headless
        │                                                  • play_media → внутри act: дожим воспроизведения
        │                                                  • self_contained → ни один пол не сработал → режим как выбрал LLM
        └──────────────────────── исполнение по режиму ────────────────────────┘
```

- **Уровень 1 (режим)** — это СУЖДЕНИЕ (сложность/неоднозначность/«знаю ли я ответ»), embeddings его не заменяют; их вклад — как ПРАЙОР (бандит) и few-shots, финальное решение за LLM.
- **Уровень 2 (intent)** — это СОДЕРЖАНИЕ запроса (нужен веб/руки/медиа), оно определяется семантикой → embeddings тут естественны и заменяют русско-регэкспы (любой язык). web_grounding — единственный сигнал, что может перебить режим (на act), потому что это анти-галлюцинационный пол.

## Слои

| Слой | Файлы | Суть |
|---|---|---|
| Когниция / мета-контроль | `agent.py` (goal/reflexion/act/reason/decompose/step/synthesize ноды) | **6 типов мышления** (fast/reason/act/deliberate/heavy/clarify; act = «System 1 с руками»: прямое действие без декомпозиции, тяжёлый пайплайн — только когда прямого действия не хватает) + **reflexion-grounding** (оценка «могу ли достоверно ответить сам» → заземление, анти-галлюцинация); целеполагание, декомпозиция, по-пунктовое исполнение |
| Память | `memory/store.py` (SQLite), `embedder.py`, `vector_index.py` (TurboVec), `feedback.py`, **`memory_tools.py`**, **`interaction.py`** | эпизоды/факты(+тэги)/выводы/цели/саммари + граф-рёбра; **УСЛОВНЫЙ recall** (`recall_scored`+гейт `recall_gate`: персона всегда, ассоциативная память по релевантности — «recall не всегда»); **GraphRAG-lite** (`_densify_fact` fact↔fact по cosine + `_graph_boost` spreading-activation от релевантных эпизод-сидов, per-user, PII-контейнмент); запрос эмбеддится ОДИН раз (qvec прокидывается, не N HTTP-вызовов); implicit feedback. **Журнал взаимодействий**: HITL/clarify переживают прогон → эпизод + harvest без LLM. **Память-как-tool (3 яруса)**: `search_memory` / `recall_history` / `note_to_self` |
| Маршрутизация интентов | **`intent.py`** · `eval/route_eval.py` | **универсальный embedding-kNN роутер** (любой язык): кодбук маршрутов {web_grounding/physical_browser/play_media/self_contained}, cosine-kNN; заменяет русско-регэксп-костыли (web-грунтинг=UNION регэксп∨классификатор — пол не ослаблен; physical/play=классификатор+регэксп-fallback). Переиспускает query-эмбеддинг из recall (0 лишних вызовов), ФИКСИРОВАН по модели (тег+инвалидация). **Растёт из фидбек-лупа** (валидированный прогон→маршрут); per-label порог (тюнинг по confusion). КОРПУС pos/neg (`route_examples.db`) для будущего обучения локального head. Стат-оценка route_eval: 410 кейсов, **94%** |
| Навыки | `tools/skill_creation.py`, `skills/*`, **`retrieval.py`** | реестр, защита core, автосинк; **ToolSearch** (BM25S-retrieval навыков при росте библиотеки); `web_search` с контекстным инжинирингом (trafilatura→чанки→BM25S→vector-rerank, полную страницу не кормит); `device_control`; **`browser_control`** (+`browser_session.py`): структурные ДЕЙСТВИЯ в браузере — снапшот DOM-элементов с номерами → клик/ввод по номеру; видимый Chromium с постоянным профилем (логины живут), `browser_see` read-only, действия под HITL |
| База знаний юзера | **`knowledge_base.py`** · **`lightrag_engine.py`** | ДВА яруса: (1) ГЛОБАЛЬНАЯ БЗ — персональные документы в иерархии папок, граф на **настоящем LightRAG** (lightrag-hku: сущности+связи, гибридный multi-hop retrieval), BM25-фолбэк без ключа; (2) СЕССИОННЫЕ файлы (ярус 3, tmp/<session_id>) — мультимодальные (pdf/image/audio/video), чистятся в конце. **AutoRAG**: recall авто-подмешивает релевантные куски БЗ+сессии через ДЕШЁВЫЙ BM25 (на каждый запрос; без LLM/эмбеддинг-трат), с провенансом «свои данные» И `sanitize_tool_output` (отравленный документ — данные, не команды); глубокий LightRAG-граф — за тулами `search_knowledge_base`/`search_attached_files`, когда агент сам решает копать; флаг `own_docs` в state глушит мнимый clarify |
| Способности-инструменты | **`research.py`** · **`compute.py`** · **`media.py`** · **`mcp_client.py`** | дисциплинированный **research** (план под-вопросов→поиск+сниппеты+чтение→ВЕРИФИКАЦИЯ факта→синтез, зависимая цепочка); **вычислительный слой** `python_exec` (точный счёт в песочнице — rlimits/kill); **vision-чтение фигур PDF** `read_pdf_figures` (рендер→vision, гейт по наличию PDF); **data-MCP само-расширение** `try_connect_discovered` (домен→discover→фильтр релевантности→первый ЖИВОЙ remote-MCP; movie/finance/weather подключаются живьём) |
| Самообучение / амортизация | `improve/`, **`habits.py`**, **`bandit.py`**, **`collective.py`**, `memory/store.py: recipes` | forward-харвест few-shots (глоб+**пер-юзер**, двухъярусно с baseline); backward: дифф-credit-assignment → per-node gradients → оптимизация промптов; **per-user backward** (`graph_backward_user`: уроки из неудач юзера → его few-shots); **измеримый accept/revert** (прогон ДО/ПОСЛЕ на кейсах) → ParamStore; **привычки** (`habits.py`: k похожих успешных дорогих прогонов → факт-директива → router создаёт навык → привычка закрывается ✅); **бандит-прайор режима** (`bandit.py`: Beta/Thompson по похожим эпизодам юзера, видит и НЕУДАЧИ — в few-shots их нет; прайор в memory_context reflexion, не диктат) |
| Трейсинг/диагностика | `tracing/` | спаны по нодам (data/traces.db), самодиагностика, ротация |
| Безопасность | `utils_validation.py` (AST-гейт), `utils.py` (песочница-подпроцесс), `hitl.py` (human-in-the-loop), **`improve/safety.py`** | генерируемый код: AST-запреты + smoke в изолированном процессе (rlimits/kill); side-effect тулы — подтверждение, deny by default; **анти-injection в выводах тулов/MCP/поиска** (`sanitize_tool_output`); **анти-PII пол** (`strip_ungrounded_pii` режет выдуманные email, числа не трогает; `redact_pii` в коллективных рецептах) — «не разглашать» = близнец «не выдумывать»; запреты обучения (не менять архитектуру/промпты, не учиться на взломе) |
| Внешнее | `external/context.py` | контекст A2A/MCP в состоянии (слот + плумбинг) |
| Обслуживание | `maintenance/dep_update.py` | безопасный авто-апдейт зависимостей с health-check и откатом |
| Интерфейсы | `main.py` (REPL), `bot.py` (Telegram), `server.py` (FastAPI) | общий граф + общая память |

## Архитектурный принцип: амортизированный агент

У известных паттернов (ReAct, plan-execute, multi-agent) предельная стоимость задачи
~постоянна. Здесь каждый успешный прогон оставляет артефакт, делающий похожие задачи
ДЕШЕВЛЕ — лестница компиляции опыта: эпизод → few-shot → **рецепт** (план+навыки;
`memory/store.py: recipes`) → привычка (`habits.py`) → навык (код). Похожая задача:
селектор берёт навыки из рецепта БЕЗ LLM-вызова; при sim≥0.7 decompose тоже БЕЗ LLM
(план из рецепта); win/lose-трекинг, проигрывающий рецепт самоудаляется. Исполнение —
лестница с проверяемой эскалацией (act → deliberate → heavy; вверх только по
заземлённому провалу).

**Эмпирика** (`scripts/amortize_bench.py`: один список задач, cold vs warm проход одного
user_id): тёплый проход **−13% токенов при росте
качества conf 78%→98%** (проваленная холодная задача 18% решена на 95%). Ключевой урок,
добытый отрицательными прогонами №1/№3: артефакт опыта должен **ЗАМЕНЯТЬ LLM-работу**
(zero-LLM селектор/декомпозиция), а не аннотировать её — хинты/few-shots/прайоры раздувают
контекст всех вызовов и покупают только надёжность. Оговорки: n=4, время шумит латентностью
API, confidence — самооценка валидатора.
**Коллективный ярус** (`collective.py`): проверенный личный рецепт (winrate-гейт) →
best-practice инсталляции с отпечатком профиля источника; похожим юзерам — рекомендация
(запрос-сходство + профиль-гейт), личное всегда приоритетнее, отрава/дрейф отсеиваются
(инъекции не промоутятся, проигрывающий глобальный рецепт самоудаляется).

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
- `python scripts/amortize_bench.py` — проверка тезиса амортизации (платный живой прогон).
- `python -m src.eval.route_eval` — стат-оценка universal intent-роутера (410 размеченных мультиязычных кейсов).
- `python scripts/gaia_resilient.py N --jsonl <path>` — GAIA held-out отказоустойчиво (переживает нативный краш, резюме по JSONL).
- REPL: `/kb add|ls|mkdir|find` — база знаний (граф LightRAG, с прикидкой цены и HITL); `/attach <файл>` — вложение сессии (tmp, чистится).

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
- Оркестрация = выбор 1 из 6 фикс-путей; лестница эскалации (act→deliberate) даёт первую динамику, но свободная композиция когнитивных модулей — дальше.
- **History-masking** длинного ReAct-контекста (старые наблюдения → заглушки) — отложено: историей сообщений владеет LangGraph `create_agent`, маскинг там = хрупкий хак. Сейчас: сжатие вывода тула (cap) + urllib-first чтение страниц.
- **LightRAG** работает для БЗ документов юзера (`knowledge_base.py`); граф-RAG для ГЛОБАЛЬНОЙ памяти (эпизоды/факты) — в очереди; сейчас recall recency+relevance+importance + TurboVec-ANN.
- Амортизация: статистика n=4 (нужна серия с медианами); LLM-генерализация рецептов перед коллективным промоушеном (privacy в мульти-юзер деплое); наследование сильных MCP через before/after-сравнение.
