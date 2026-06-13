# Подключение и настройка

## 0. База
```bash
uv sync
.venv/bin/python -m playwright install chromium   # для веб-поиска (cloakbrowser)
```
`.env`:
```
OPEN_ROUTER_API_KEY=sk-or-...        # обязателен — LLM И эмбеддинги через OpenRouter
SEARXNG_URL=http://localhost:8080    # опц. — приватный свежий поиск
TELEGRAM_BOT_TOKEN=...               # опц. — Telegram-бот
# OPENAI_API_KEY=...                 # опц. — альтернатива OpenRouter для эмбеддингов
```
Запуск:
```bash
.venv/bin/python main.py             # REPL (rich + редактирование строки/история)
.venv/bin/python bot.py              # Telegram-бот
uvicorn src.server:app --port 8000   # HTTP API + веб-GUI на http://localhost:8000/
.venv/bin/python desktop.py          # нативное окно с чат-GUI (uv sync --group gui)
```
**GUI**: сервер отдаёт чат-интерфейс на `/` (открой в браузере), либо `desktop.py` —
нативное окно ОС (системный webview, без Electron). Тонкий клиент + мозг: окно говорит
с `/chat` локального сервера.

## 0.1. Кроссплатформенность и упаковка

**Установка на любой ОС (Windows / macOS / Linux)** — через `uv`:
```bash
uv sync                              # macOS-only deps (pyobjc) гейтятся сами → ставится везде
uv run self-extension-agent          # консольная команда (REPL); с аргументом — one-shot
uv run self-extension-agent "посчитай 2^10"   # автономный прогон одной задачи
```
> macOS-специфичные навыки (AX/keystroke управления окнами) на Windows/Linux просто не
> грузятся — ядро (поиск/анализ/память/браузер) работает кроссплатформенно.

**Нативный бинарь (.exe / .app / бинарь), без Python у пользователя** — PyInstaller:
```bash
uv sync --group package              # поставить pyinstaller
python scripts/build_binary.py       # собрать под ТЕКУЩУЮ ОС → dist/self-extension-agent[.exe]
```
PyInstaller не кросс-компилирует: для всех трёх ОС сразу — CI
`.github/workflows/build.yml` (матрица ubuntu/macos/windows → артефакты). При первом запуске
бинарь кладёт дефолтный `config.yml` рядом с собой; данные (`data/`, `config.local.yml`)
персистятся в рабочей папке.

**Провайдер и ключ без `.env`** — в REPL `/config` → пункт «Провайдер/ключ»: ввод API-ключа и
`base_url` (любой OpenAI-совместимый endpoint) с живой валидацией; сохраняется в
`config.local.yml`. env-ключ (`OPEN_ROUTER_API_KEY`) имеет приоритет.

## 1. Работа с приложениями ПК «без костылей»

### app_control — скриптуемые приложения (Mail/Safari/Calendar/Notes), AppleScript
**System Settings → Privacy & Security → Automation** → разреши Terminal (или iTerm/python)
управлять нужными приложениями. Первый вызов вызовет системный запрос — подтверди.

### ax_control — любые приложения (Accessibility-дерево)
**System Settings → Privacy & Security → Accessibility** → нажми `+` и добавь свой
терминал (Terminal.app / iTerm) или интерпретатор Python. Без этого `read_ui` вернёт
подсказку. После выдачи — `read_ui("Notes")` отдаёт реальное дерево элементов.

### device_control — общие действия (скролл/клавиши/уведомления/TTS)
Тоже требует **Accessibility** (как выше). `send_telegram`/`scroll`/`type_text` шлют
клавиши в активное окно.

> Безопасно тестировать: `export AGENT_DRY_RUN=1` — действия с устройством не выполняются
> (только логируются), чтение остаётся рабочим.

## 2. Телефон (Android) — phone_control
```bash
brew install android-platform-tools          # adb
```
На телефоне: **Настройки → О телефоне → 7 раз тапнуть «Номер сборки»** (вкл. режим
разработчика) → **Параметры разработчика → Отладка по USB (вкл.)**.
Подключи по USB → на телефоне подтверди «Разрешить отладку» → проверь:
```bash
adb devices        # должно показать устройство как "device"
```
По Wi-Fi (опц.): `adb tcpip 5555` затем `adb connect <ip_телефона>:5555`.
Дальше агент сам: `phone_ui` (дерево экрана) → `phone_tap`/`phone_type`/`phone_open_app`.
iOS — сложнее (нужен Mac + WebDriverAgent/Appium), пока не подключено.

## 3. Свежий приватный поиск — SearXNG (опц., рекомендуется)
```bash
docker run -d -p 8080:8080 --name searxng searxng/searxng
export SEARXNG_URL=http://localhost:8080
```
Без него поиск идёт через cloakbrowser (stealth) → urllib (фолбэк).

## 4. MCP (внешние инструменты)
- **Авто** к доверенным (allowlist в `src/mcp_client.py`, напр. `fetch`) — через `uvx`
  (ничего ставить не надо, сервер скачается сам при первом вызове).
- **Discovery**: агент ищет сервер под задачу в официальном реестре
  (`registry.modelcontextprotocol.io`); найденное — **предложения**, подключаются только
  после подтверждения (`approve_server`) или при `config mcp.auto_trust_discovered=true`.
- node/npx серверы требуют Node.js (`brew install node`); Python-серверы идут через uvx.

## 4b. Локальные модели — Ollama (бесплатно по токенам)
```bash
brew install ollama
# ВАЖНО: промпты агента ~6–8k токенов → поднять контекст при старте сервера
OLLAMA_CONTEXT_LENGTH=16384 ollama serve &
ollama pull gpt-oss:20b           # MoE ~3.6B-active — влезает в 24ГБ M4 Pro (самый слабый)
ollama pull qwen2.5-coder:7b      # для кода/исполнения
ollama pull nomic-embed-text      # эмбеддинги (если включены)
```
> 48ГБ+ RAM → можно `qwen3:30b-a3b` (мощнее). 24ГБ → `gpt-oss:20b`.
> Без `OLLAMA_CONTEXT_LENGTH` Ollama режет контекст (дефолт мал) → агент деградирует.

**Чтобы НЕ грелся (M4 Pro):**
- macOS **Low Power Mode** (Настройки → Аккумулятор) — главный и самый простой рычаг, заметно холоднее.
- Один загруженный модель за раз: `OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 ollama serve`
  (иначе fast+code модели висят в памяти разом). Можно сделать `code_model: gpt-oss:20b` (одна модель на всё).
- Меньше вызовов: в `config.yml` — `agent.consensus_validation: false` (2 судьи → 1),
  `improve.auto: false`, поменьше `memory.recall_budget_chars` (короче промпты = меньше счёта).
- Если всё равно тёплый — возьми мелкую модель: `qwen3:4b` / `llama3.2:3b` (холоднее и быстрее, но проще).
Переключение — одна строка в `config.yml`:
```yaml
provider: ollama   # было: openrouter
```
Всё (граф, self-learning, эмбеддинги) пойдёт локально. Имена моделей — в секции `ollama:`.
Назад в облако — `provider: openrouter`.

## 5. Семантическая память (эмбеддинги + TurboVec)
В `config.yml`: `memory.embeddings: true` — recall станет семантическим (через OpenRouter,
тем же ключом; модель — `memory.embedding_model`).

## 6. Самообучение
`config.yml → improve.auto: true` — backward по графу запускается по деградации качества
и при неактивности (idle). Вручную: `python -m src.improve --graph`.
Диагностика/трейсы: `python -m src.tracing`. Обновление зависимостей: `python -m src.maintenance`.

## Сводка разрешений macOS
| Возможность | Что включить |
|---|---|
| ax_control / device_control | Privacy → **Accessibility** (+ Terminal/python) |
| app_control (AppleScript) | Privacy → **Automation** (Terminal → приложения) |
| screencapture | Privacy → **Screen Recording** (если нужен скрин) |
| phone_control | `adb` + USB-отладка на телефоне |
