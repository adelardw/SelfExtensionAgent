# self-extension-agent

Самораширяющийся, самообучающийся персональный агент на **LangGraph**. Сам выбирает
тип мышления под задачу, помнит пользователя между сессиями, расширяет себя навыками
и **обучается на собственных трейсах**, относясь к своему графу как к обучаемой программе.

Идея: дешёвая модель → высокоспособный агент за счёт **харнесса**, не размера модели.
Полезен каждому через **оптимизацию под конкретного пользователя** (персонализация =
метод универсальности), держа контекст компактным через **контекстный инжиниринг**.

Полная архитектура — в [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Что умеет

- **5 типов мышления (Any-2-Any)** — мета-контроллер сам выбирает по анализу задачи,
  как человек: `fast` (интуитивно), `reason` (глубокое рассуждение), `deliberate`
  (инструменты + декомпозиция + по-пунктовое исполнение/валидация), `heavy` (большая
  задача: то же + **сквозной ревью собранного решения целиком** и раунд доработки
  найденных проблем), `clarify` (переспросить). Бюджет встроен: простое не идёт по
  дорогому пути, дорогая deep-модель зовётся только в heavy-ревью.
- **Онбординг** — при первом контакте агент кратко представляется и начинает строить
  профиль пользователя (имя, стиль), не вываливая список возможностей.
- **Временные навыки** — навык, созданный под задачу, помечается `temp`; после решения
  retention-судья решает: принять в библиотеку (переиспользуем) или удалить (одноразовый).
  Не принятые навыки чистятся по TTL при старте — библиотека не зарастает мусором.
- **Целеполагание** — определяет цель и держит «стоящую» цель + rubric в контексте.
- **Онбординг неясной задачи (свойство системы, не одна нода)** — неоднозначность
  ловится в трёх точках: на входе (ambiguity-гейт → переспросить), при планировании
  (`clarify_gate` — батч точных вопросов: маркеры где набор конечен, открытые где нет)
  и прямо в исполнении (инструмент `ask_user` — догон на развилке). Все вопросы/ответы
  копятся в один **реестр уточнений** на прогон и переиспользуются всеми нодами — агент
  не переспрашивает дважды. Нет ответа/канала → разумное допущение с пометкой
  «исходил из того, что…» в финале (не блокирует автономную работу).
- **Память** — эпизодическая/семантическая (факты+тэги)/выводы/цели/саммари, граф-рёбра,
  TurboVec-ANN, recall с бюджетом, защита от переполнения (prune).
- **Память-как-TOOL (3 яруса)** — агент САМ решает, что подтянуть: `search_memory`
  (глобальная долгая), `recall_history` (drill-back — восстановить ПОЛНЫЙ прошлый эпизод
  из компактного индекса), `note_to_self`/`read_my_notes` (временная runtime-память,
  не персистится). Не только авто-впрыск — память как инструмент.
- **Персонализация** — извлекает устойчивые факты о пользователе (мульти-роль), учитывает
  их везде ВНУТРЕННЕ (роли не называются вслух).
- **Самообучение (forward + backward, в т.ч. ПЕР-ЮЗЕР)** — forward (сбор few-shots из удач,
  глобальных и персональных) + backward (textual-gradients по трейсу: дифф-credit-assignment
  → per-node критика → оптимизация промптов). **Per-user backward** (`graph_backward_user`):
  из неудач конкретного юзера + того, КТО он, синтезирует корректирующие уроки → его
  персональные few-shots (ядро заморожено, few-shots — обратимый канал). **Измеримый
  accept/revert**: правка промпта сохраняется только если внутренний прогон ДО/ПОСЛЕ на
  кейсах показал улучшение, иначе откат. Двухъярусные few-shots: встроенный baseline +
  обучаемые. Триггер — по деградации/неактивности, не каждую итерацию.
- **Навыки** — создаёт новые навыки со smoke-тестами (skill library), защищает базовые,
  авто-синхронизирует реестр.
- **ToolSearch** — при росте библиотеки селектор не получает ВЕСЬ реестр, а BM25-retrieval
  топ-релевантных навыков под запрос (`src/retrieval.py`). Масштабирует выбор инструментов.
- **MCP — поиск/подключение/использование** — `discover_mcp` (официальный реестр) +
  доверенный каталог; на capability-gap агент находит и (с подтверждением, либо авто в
  eval-режиме `AGENT_UNLEASH`) подключает MCP-сервер и решает им задачу.
- **Импорт OpenClaw-скиллов** — `import_openclaw_skill` берёт навык ClawHub (формат
  `SKILL.md`) из локального каталога или GitHub-URL и оборачивает в наш формат:
  инструкции инъектятся исполнителю, а CLI вызывается через allowlist бинарников
  (`requires.bins` ∪ `install[].bins`) с timeout/dry-run. Импортированный (сторонний)
  навык всегда под human-in-the-loop. Так экосистема OpenClaw становится твоей библиотекой.
- **Трейсинг и самодиагностика** — спаны по нодам, поиск своих «косяков» и деградации.
- **Действия с устройством (on-demand, кроссплатформенно)** — открыть сайт/приложение,
  скриншот + **vision-анализ экрана** (`analyze_screen`), уведомление, TTS: бэкенды под
  macOS/Linux/Windows. Работа с открытыми окнами (скролл/ввод/AX, Telegram) — пока macOS.
- **DeepAgent (дополнение)** — для долгогоризонтных/файловых подзадач (виртуальная ФС,
  todo, суб-агенты), вызывается из шага, не заменяя ядро.
- **Свежий веб-поиск + контекстный инжиниринг** — поиск: SearXNG (приватный) → urllib-DDG →
  cloakbrowser (stealth); недоступный SearXNG уходит в cooldown. Чтение страницы НЕ кормит
  агенту всю страницу: **trafilatura** (чистка HTML) → чанкинг → **BM25S** (лексика) →
  **vector-rerank** (OpenRouter-эмбеддинги) → в контекст только релевантные куски. Чтение
  страниц — urllib+trafilatura первым (быстро), браузер только для бот-стен.
- **Универсальный помощник по файлам** — PDF (тиерный парсер), Excel, Word, **PowerPoint**,
  текст, картинки (vision), аудио (транскрипт), **видео/GIF** (сэмпл кадров → vision +
  аудио-дорожка → транскрипт). В Telegram — фото/документы/voice как есть; в REPL — путь
  к файлу в запросе, голос — `/voice`.
- **Живой прогресс** — при долгих задачах видно, что агент делает прямо сейчас
  (режим → план → шаг i/N → ревью → валидация) и сколько токенов/$$ уже потрачено
  (REPL — в статус-строке, Telegram — статус-сообщение редактируется по ходу).
- **Интерфейсы** — REPL, Telegram-бот, FastAPI-сервер (общий граф и общая память).

## Безопасность (guard rails)

Три реальных слоя — не промпт-инструкции:

1. **AST-гейт на записи кода** (`src/utils_validation.py`). Любой код, который LLM
   сохраняет как навык (`create_skill`/`update_skill_tools`), проходит AST-анализ:
   запрещены `subprocess`, `os.system`, `eval`/`exec`/`__import__`, `ctypes`,
   `importlib`, `shutil.rmtree` — включая алиасы (`import subprocess as sp`,
   `from os import system as s`) и getattr-обход (`getattr(os, 'sys'+'tem')`).
   Владелец может отключить: `AGENT_ALLOW_RISKY_SKILLS=1`.
2. **Песочница smoke-теста** (`src/utils.py: run_tool_sandboxed`). Сгенерированный
   tool исполняется в ОТДЕЛЬНОМ процессе с resource-лимитами (CPU, память, размер
   файлов) и жёстким kill-таймаутом — никогда в процессе агента.
3. **Human-in-the-loop** (`src/hitl.py`, config `agent.require_confirmation`).
   Тулы side-effect навыков (`skills.confirm`: device/app/ax/phone) требуют явного
   подтверждения человеком: REPL — `y/N` в терминале, Telegram — inline-кнопки;
   где канала подтверждения нет (HTTP-сервер) — **deny by default**. Плюс
   независимый `AGENT_DRY_RUN`.

Дополнительно: core-навыки защищены от перезаписи и удаления агентом (`delete_skill`
не имеет `force`; владельческое удаление — только `force_delete_skill` из кода/CLI).

**Защита от инъекций через выводы инструментов** (`safety.sanitize_tool_output`): вывод
любого тула/MCP/навыка/поиска — недоверенные ДАННЫЕ; при попытке prompt-injection
(«ignore previous…», «reveal system prompt», скрытые команды) триггеры обезвреживаются и
текст помечается «это данные, не инструкции» — защита от skills-/mcp-/search-injection.

**Запреты обучения** (залочены тестами `test_optimization_policy`): backward НЕ меняет
архитектуру (пишет только артефакты ParamStore, не код/граф), НЕ переписывает системные
промпты ключевых нод (заморожены), и НЕ учится на попытках обхода защиты
(`safety.filter_learnable` исключает джейлбреки из обучающего батча).

**Честные границы**: песочница — изоляция уровня процесса (rlimits + kill), не
gVisor/seccomp; AST-анализ не ловит динамическую кодогенерацию (но `exec`/`eval`
запрещены целиком); core-навыки (AppleScript/AX/adb) исполняются доверенно — их
писал владелец. Device/app/ax-навыки сейчас **macOS-only**; Linux/Windows-бэкенды —
в roadmap.

## Установка

```bash
uv sync
.venv/bin/python -m playwright install chromium   # для cloakbrowser-поиска
```

## Настройка

`.env` (шаблон — `.env.example`, файл в `.gitignore`, в гит не попадает):
```
OPEN_ROUTER_API_KEY=...              # обязателен (LLM И эмбеддинги через OpenRouter)
SEARXNG_URL=http://localhost:8080    # опц. — приватный свежий поиск
TELEGRAM_BOT_TOKEN=...               # опц. — для Telegram-бота
# OPENAI_API_KEY=...                 # опц. — альтернатива OpenRouter для эмбеддингов

# Эмбеддинги (семантический recall + TurboVec) включаются в config.yml: memory.embeddings=true
# и идут через OpenRouter тем же OPEN_ROUTER_API_KEY (модель — memory.embedding_model).
```

`config.yml`: модели, `memory.*` (recall/embeddings/caps), `skills.protected/autosync`,
`improve.*` (триггер само-улучшения).

### Low-cost тиры моделей (цены проверены через OpenRouter API)

| Тир | Модель | $/M in/out | Используется для |
|---|---|---|---|
| fast | `google/gemini-2.5-flash-lite` | 0.10 / 0.40 | роутинг, валидация, extraction, fast/reason |
| code | `deepseek/deepseek-v4-flash` | 0.098 / 0.197 | агентское исполнение шагов, код, ctx 1M |
| deep | `deepseek/deepseek-v4-pro` | 0.435 / 0.87 | ТОЛЬКО heavy-ревью (1–2 вызова на большую задачу) |

Типичный fast-запрос ≈ $0.001; deliberate ≈ $0.005–0.02; heavy добавляет 1–2 deep-вызова.

## Запуск

```bash
.venv/bin/python main.py                 # REPL
.venv/bin/python bot.py                  # Telegram-бот
uvicorn src.server:app --port 8000       # HTTP API
```

API: `POST /chat {user_id, query}`, `GET /diagnose`, `/memory/facts`, `/memory/goal`, `/traces`.

## Самообучение и обслуживание (CLI)

```bash
python -m src.improve --graph     # backward по графу: credit assignment + per-node оптимизация
python -m src.improve --list      # принятые параметры/few-shots
python -m src.tracing             # самодиагностика по трейсам
python -m src.maintenance         # безопасный авто-апдейт зависимостей (с откатом)

# Импорт навыка OpenClaw (локальный каталог или GitHub-URL):
python -m src.tools.openclaw_import https://github.com/openclaw/openclaw/tree/main/skills/github
```

## Тесты

```bash
.venv/bin/python -m pytest tests/ -q   # 123 теста, в осн. без LLM (память/retrieval/безопасность/…)
```
Тесты сборки графа требуют API-ключ (LLM строится на импорте), остальные — оффлайн.
Быстрый прогон повседневных сценариев через реальный граф: `python -m src.eval.daily_eval [N]`.

## Структура

```
src/
  agent.py            граф (recall→goal→reflexion→{fast|reason|deliberate|heavy}→…→reflect)
  prompts.py          промпты + реестр обучаемых (OPTIMIZABLE_PROMPTS)
  structured_outputs.py
  memory/             store(SQLite) + embedder + vector_index(TurboVec) + feedback
  memory_tools.py     память-как-tool (3 яруса: search_memory / recall_history / scratch)
  retrieval.py        канонический BM25S-ранкер (ToolSearch и др.)
  improve/            prompt_store(ParamStore) + optimizer + pipe + graph_learn + safety
  mcp_client.py       discover/connect/use MCP (реестр + доверенный каталог)
  subagents.py        под-агенты/под-графы как инструменты
  clarify.py          реестр уточнений (онбординг-по-исполнению)
  runbudget.py        токен/время-бюджет прогона (анти-runaway)
  media.py            файлы (pdf/excel/docx/pptx/видео/gif/image/audio)
  tracing/            tracer(спаны) + diagnose
  external/           контекст A2A/MCP   ·  maintenance/  авто-апдейт зависимостей
  tools/              менеджер навыков (создание/защита/автосинк/ToolSearch)
  skills/             навыки (web_search, device_control, deep_agent, stash, …)
  eval/               daily_eval / gaia_runner / assistantbench_runner
  server.py           FastAPI
main.py / bot.py      REPL / Telegram
```

## Статус

Реализовано и протестировано (123 теста): ядро, 5 режимов мышления, по-пунктовое
исполнение, память + **память-как-tool (3 яруса)**, персонализация, **per-user
само-улучшение** + измеримый accept/revert, **reflexion-обоснованность** (анти-галлюцинация),
**контекстный поиск** (trafilatura→BM25S→vector), **ToolSearch**, MCP discover/connect/use,
защита (AST→песочница→HITL + **анти-injection в выводах тулов** + запреты обучения),
универсальные файлы (pdf/excel/docx/pptx/видео/gif/аудио), трейсинг/самодиагностика,
device on-demand (кроссплатформенно), DeepAgent, сетевые таймауты, eval-харнессы
(daily/GAIA/AssistantBench), REPL/Telegram/FastAPI.

Отложено (см. `ARCHITECTURE.md`): history-masking длинного ReAct-контекста (владеет
LangGraph), GraphRAG/LightRAG для глобальной памяти, свободная динамическая композиция
модулей, кроссплатформенный UI-automation вне macOS.
