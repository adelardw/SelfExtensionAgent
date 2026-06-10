# self-extension-agent

Самораширяющийся, самообучающийся персональный агент на **LangGraph**. Сам выбирает
тип мышления под задачу, помнит пользователя между сессиями, расширяет себя навыками
и **обучается на собственных трейсах**, относясь к своему графу как к обучаемой программе.

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
- **Персонализация** — извлекает устойчивые факты о пользователе, учитывает их везде.
- **Самообучение** — forward (сбор few-shots из удач) + backward (textual-gradients по
  трейсу: дифф-credit-assignment → per-node критика → оптимизация промптов → валидация).
  Запускается **не каждую итерацию**, а по деградации качества или при неактивности.
- **Навыки** — создаёт новые навыки со smoke-тестами (skill library), защищает базовые,
  авто-синхронизирует реестр.
- **Импорт OpenClaw-скиллов** — `import_openclaw_skill` берёт навык ClawHub (формат
  `SKILL.md`) из локального каталога или GitHub-URL и оборачивает в наш формат:
  инструкции инъектятся исполнителю, а CLI вызывается через allowlist бинарников
  (`requires.bins` ∪ `install[].bins`) с timeout/dry-run. Импортированный (сторонний)
  навык всегда под human-in-the-loop. Так экосистема OpenClaw становится твоей библиотекой.
- **Трейсинг и самодиагностика** — спаны по нодам, поиск своих «косяков» и деградации.
- **Действия с устройством (on-demand)** — открыть сайт/приложение, скриншот, уведомление,
  TTS, скролл открытого браузера, ввод текста, отправка в Telegram.
- **DeepAgent (дополнение)** — для долгогоризонтных/файловых подзадач (виртуальная ФС,
  todo, суб-агенты), вызывается из шага, не заменяя ядро.
- **Свежий веб-поиск** — SearXNG (приватный) → cloakbrowser (stealth) → urllib;
  недоступный SearXNG уходит в cooldown (без спама в лог).
- **Вложения и голос** — картинки (vision той же fast-моделью: текст/таблицы/UI с
  изображения), файлы (текст инлайнится, бинарное — навыками), голосовые сообщения
  (расшифровка fast-моделью, ffmpeg для ogg). В Telegram — фото/документы/voice как есть;
  в REPL — упомяни путь к файлу в запросе, голос — команда `/voice` (запись до Enter).
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
.venv/bin/python -m pytest tests/ -q   # smoke-набор без LLM (память/ParamStore/credit assignment)
```
Тесты сборки графа требуют API-ключ (LLM строится на импорте), остальные — оффлайн.

## Структура

```
src/
  agent.py            граф (recall→goal→reflexion→{fast|reason|deliberate}→…→reflect)
  prompts.py          промпты + реестр обучаемых (OPTIMIZABLE_PROMPTS)
  structured_outputs.py
  memory/             store(SQLite) + embedder + vector_index(TurboVec) + feedback
  improve/            prompt_store(ParamStore) + optimizer(TextGrad/Reflexion) + pipe + graph_learn
  tracing/            tracer(спаны) + diagnose
  external/           контекст A2A/MCP
  maintenance/        авто-апдейт зависимостей
  tools/              менеджер навыков (создание/защита/автосинк)
  skills/             навыки (web_search, device_control, deep_agent, …)
  server.py           FastAPI
main.py / bot.py      REPL / Telegram
```

## Статус

Ядро, память, персонализация, 4 режима мышления, по-пунктовое исполнение, self-learning
(forward+backward, триггер по деградации/неактивности), трейсинг, защита/переполнение,
device on-demand, DeepAgent, SearXNG, сервер — реализованы и протестированы. Отложено
(см. `ARCHITECTURE.md`): vision-анализ скриншота, реальный MCP-клиент с human-gate,
свободная композиция модулей, gradient вдоль рёбер трейса.
