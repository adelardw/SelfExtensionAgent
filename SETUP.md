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
uvicorn src.server:app --port 8000   # HTTP API (/chat, /diagnose, /memory/*, /traces)
```

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
