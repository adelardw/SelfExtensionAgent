# self-extension-agent

Самораширяющийся, самообучающийся персональный агент на **LangGraph**. Сам выбирает
тип мышления под задачу, помнит пользователя между сессиями, расширяет себя навыками
и **обучается на собственных трейсах**, относясь к своему графу как к обучаемой программе.

Полная архитектура — в [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Что умеет

- **4 типа мышления (Any-2-Any)** — мета-контроллер сам выбирает по анализу задачи:
  `fast` (интуитивно), `reason` (глубокое рассуждение), `deliberate` (инструменты +
  декомпозиция + по-пунктовое исполнение/валидация), `clarify` (переспросить). Бюджет
  встроен: простое не идёт по дорогому пути.
- **Целеполагание** — определяет цель и держит «стоящую» цель + rubric в контексте.
- **Память** — эпизодическая/семантическая (факты+тэги)/выводы/цели/саммари, граф-рёбра,
  TurboVec-ANN, recall с бюджетом, защита от переполнения (prune).
- **Персонализация** — извлекает устойчивые факты о пользователе, учитывает их везде.
- **Самообучение** — forward (сбор few-shots из удач) + backward (textual-gradients по
  трейсу: дифф-credit-assignment → per-node критика → оптимизация промптов → валидация).
  Запускается **не каждую итерацию**, а по деградации качества или при неактивности.
- **Навыки** — создаёт новые навыки со smoke-тестами (skill library), защищает базовые,
  авто-синхронизирует реестр.
- **Трейсинг и самодиагностика** — спаны по нодам, поиск своих «косяков» и деградации.
- **Действия с устройством (on-demand)** — открыть сайт/приложение, скриншот, уведомление,
  TTS, скролл открытого браузера, ввод текста, отправка в Telegram.
- **DeepAgent (дополнение)** — для долгогоризонтных/файловых подзадач (виртуальная ФС,
  todo, суб-агенты), вызывается из шага, не заменяя ядро.
- **Свежий веб-поиск** — SearXNG (приватный) → cloakbrowser (stealth) → urllib.
- **Интерфейсы** — REPL, Telegram-бот, FastAPI-сервер (общий граф и общая память).

## Установка

```bash
uv sync
.venv/bin/python -m playwright install chromium   # для cloakbrowser-поиска
```

## Настройка

`.env`:
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
```

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
