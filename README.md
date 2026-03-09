# Self-Extension Agent

Саморасширяющийся AI-агент на LangGraph, который **сам создаёт себе навыки** (инструменты) в рантайме, валидирует их через smoke-тесты и использует для выполнения задач.

---

## Архитектура

```
                              ┌─────────────┐
                              │    START    │
                              └──────┬──────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │      [1] ROUTER        │
                        │                        │
                        │  Запрос + история      │
                        │  + список навыков      │
                        │         ↓              │
                        │  create_skill ?        │
                        │  use_skills   ?        │
                        └─────┬──────────┬───────┘
                              │          │
                 route="create_skill"    route="use_skills"
                              │          │
                              ▼          │
                 ┌────────────────────┐  │
                 │ [2] CREATE_SKILLS  │  │
                 │                    │  │
                 │  ReAct-агент       │  │
                 │  + manager tools   │  │
                 │  + code_llm        │  │
                 └────────┬───────────┘  │
                          │              │
                          ▼              │
                 ┌───────────────────┐   │
                 │ [3] SGR_CREATE    │   │
                 │                   │   │
                 │  Этап 1: Статич.  │   │
                 │  ревью (LLM)      │   │
                 │                   │   │
                 │  Этап 2: Smoke    │   │
                 │  test (runtime)   │   │
                 └──┬──────┬──────┬──┘   │
                    │      │      │      │
                  FAIL   OK    GIVE UP   │
                    │      │      │      │
                    │      │      └───── ┤
                    │      │             │
                    ▼      ▼             ▼
              [retry x3]  router   ┌────────────────────┐
                                   │ [4] SKILL_SELECTOR │
                                   │                    │
                                   │  Выбор навыков     │
                                   │  из реестра        │
                                   └─────────┬──────────┘
                                             │
                                             ▼
                                   ┌────────────────────┐
                                   │ [5] PLANNING       │
                                   │                    │
                                   │  Пошаговый план    │
                                   │  выполнения        │
                                   └─────────┬──────────┘
                                             │
                                             ▼
                                   ┌─────────────────────┐
                                   │ [6] SKILL_INJECTION │
                                   │                     │
                                   │  Загрузка tools     │
                                   │  + системные промпты│
                                   └─────────┬───────────┘
                                             │
                                             ▼
                                   ┌──────────────────────┐
                                   │ [7] EXECUTION        │
                                   │                      │
                                   │  ReAct-агент         │
                                   │  + загруженные tools │
                                   │  + code_llm          │
                                   └─────────┬────────────┘
                                             │
                                             ▼
                                   ┌─────────────────────┐
                                   │ [8] VALIDATION      │
                                   │                     │
                                   │  SGR финальный      │
                                   │  confidence >= 0.7? │
                                   └──┬──────────────┬───┘
                                      │              │
                                confidence OK   confidence LOW
                                      │         (retry x2)
                                      ▼              │
                                   ┌──────┐          │
                                   │ END  │     router ◄──┘
                                   └──────┘
```

---

## Как это работает

### Два пути обработки запроса

**Путь 1 — Создание навыка** (левая ветка):
если ни один существующий навык не подходит, агент сам пишет код нового инструмента, валидирует его и добавляет в реестр.

**Путь 2 — Использование навыков** (правая ветка):
если подходящие навыки уже есть, агент выбирает их, строит план, инъектирует промпты и выполняет задачу.

### Валидация навыков (SGR Create)

Двухэтапная проверка каждого создаваемого навыка:

| Этап | Что проверяет | Как |
|------|---------------|-----|
| **Статический ревью** | Код, импорты, плейсхолдеры, @tool декораторы | LLM анализирует код |
| **Smoke test** | Реально ли работает код | Загружает модуль, вызывает @tool функцию, проверяет результат |

Автоматический отказ при обнаружении:
- Заглушки: `results = []`, `pass`, `# TODO`
- Плейсхолдеры ключей: `YOUR_API_KEY`, `REPLACE_ME`, `<api_key>`
- API с обязательной регистрацией (OpenWeatherMap, Google API)
- Рантайм ошибки: `401`, `403`, `No module named`

### Структура навыка

```
src/skills/
├── registry.json                  # Реестр всех навыков
└── weather_check/
    ├── weather_check.md           # Описание: когда и как использовать
    ├── weather_check.py           # Код с @tool функциями
    └── prompt.md                  # Системный промпт для execution-агента
```

При использовании навыка:
1. `skill_injection_node` загружает `.py` файл через `importlib`
2. `prompt.md` инъектируется в системный промпт execution-агента
3. Execution-агент получает доступ к `@tool` функциям навыка

---

## Конфигурация

```yaml
# config.yml

model:
  name: gpt-4o-mini          # Основная модель (роутер, валидация, планирование)
  temperature: 0

code_model:
  name: google/gemini-3-flash-preview   # Модель для генерации кода
  temperature: 0

agent:
  max_create_retries: 3       # Макс. попыток создания навыка
  max_global_retries: 2       # Макс. повторов при низкой confidence
  low_confidence_threshold: 0.7  # Порог принятия ответа (0.0–1.0)

checkpointer:
  backend: sqlite             # "sqlite" или "memory"
  sqlite_path: data/checkpoints.db
```

### Две модели

| Модель | Для чего | Почему |
|--------|----------|--------|
| `model` (gpt-4o-mini) | Роутинг, планирование, валидация, выбор навыков | Быстрая, дешёвая, хороша в рассуждениях |
| `code_model` (deepseek) | Генерация кода навыков, исполнение | Специализирована на коде, дешёвая |

Обе модели работают через [OpenRouter](https://openrouter.ai/).

---

## Структура проекта

```
self_extension_agent/
├── main.py                    # Точка входа, REPL, управление chat_history
├── config.yml                 # Конфигурация моделей и агента
├── pyproject.toml             # Зависимости
│
├── src/
│   ├── agent.py               # Граф: ноды, роутинг, chains, smoke test
│   ├── prompts.py             # Все промпты (на русском)
│   ├── schemas.py             # GeneralGraphState (TypedDict)
│   ├── structured_outputs.py  # Pydantic модели для structured output
│   │
│   ├── tools/
│   │   ├── __init__.py        # Экспорт: get_manager_tools, get_all_loaded_skill_tools
│   │   └── skill_creation.py  # 7 @tool функций + утилиты управления навыками
│   │
│   └── skills/                # Динамически создаваемые навыки
│       ├── registry.json
│       └── <skill_name>/
│           ├── <skill_name>.md
│           ├── <skill_name>.py
│           └── prompt.md
│
└── data/
    └── checkpoints.db         # SQLite состояние графа
```

---

## Запуск

```bash
# Установка зависимостей
uv sync

# Настройка API ключа — замените в src/agent.py на свой OpenRouter API key

# Запуск
uv run python main.py
```

### Команды REPL

| Команда | Действие |
|---------|----------|
| `exit` / `quit` / `q` | Выход |
| `new` | Новый тред (очистка истории) |
| Любой текст | Запрос к агенту |

### Пример сессии

```
Self-Extension Agent
============================================
Checkpointer: sqlite
Thread: a1b2c3d4...

> Какая погода в Москве?

[Router] create_skill — нет навыка для погоды
[Create] Создаю навык weather_check...
[SGR Static] ✓ Код валидный, нет плейсхолдеров
[SGR Smoke] ✓ get_weather("London") → "London: Clear, 15°C, ..."
[Router] use_skills — навык weather_check найден
[Skill Selector] → weather_check
[Execution] Вызываю get_weather...

============================================
Answer:
Погода в Москве: Облачно, 8°C, влажность 72%, ветер 12 км/ч

[SGR] Ответ полностью отвечает на запрос.
[Confidence] 95%

> А в Лондоне?

============================================
Answer:
Погода в Лондоне: Ясно, 15°C, влажность 55%, ветер 8 км/ч

[SGR] Ответ корректен с учётом контекста диалога.
[Confidence] 92%
```

---

## Ноды графа

### [1] Router

Решает: создать новый навык или использовать существующие.
Получает список всех навыков из реестра + историю диалога.
Короткие ответы (`да`, `нет`, `продолжай`) интерпретирует через контекст.

### [2] Create Skills

ReAct-агент с `code_llm` и manager tools.
Создаёт навык: описание + код + системный промпт.
При ретрае получает обратную связь и удаляет предыдущую версию.

### [3] SGR Create

**Этап 1** — статический ревью LLM: синтаксис, импорты, плейсхолдеры, структура.
**Этап 2** — smoke test: загрузка модуля → генерация тестового кейса → вызов @tool функции → проверка результата (таймаут 15с).
При провале — удаляет навык, отправляет feedback на retry.

### [4] Skill Selector

Выбирает подходящие навыки из реестра по запросу.

### [5] Planning

Строит пошаговый план выполнения на основе выбранных навыков.

### [6] Skill Injection

Загружает `@tool` функции навыков через `importlib`.
Собирает `prompt.md` файлы и инъектирует их в системный промпт исполнителя.

### [7] Execution

ReAct-агент с `code_llm`, загруженными tools и инъектированными промптами.
Выполняет план и генерирует финальный ответ.

### [8] Validation

SGR финальной проверки — оценивает ответ по запросу.
`confidence >= 0.7` → ответ принят.
`confidence < 0.7` → retry (макс. 2 раза).

---

## Chains и Structured Outputs

Все LLM-цепочки в глобальном скоупе:

```python
route_chain          = router_prompt         | llm.with_structured_output(RouteDecision)
sgr_create_chain     = sgr_create_prompt     | llm.with_structured_output(SGRCreateResult)
test_case_chain      = test_case_prompt      | llm.with_structured_output(SkillTestCase)
skill_selector_chain = skill_selector_prompt | llm.with_structured_output(SkillSelection)
planning_chain       = planning_prompt       | llm.with_structured_output(ExecutionPlan)
validation_chain     = validation_prompt     | llm.with_structured_output(ValidationResult)
```

| Модель | Назначение |
|--------|-----------|
| `RouteDecision` | Маршрут: `create_skill` / `use_skills` |
| `SkillSelection` | Список выбранных навыков |
| `ExecutionPlan` | Пошаговый план |
| `SGRCreateResult` | Валидация навыка: is_valid, confidence, issues |
| `SkillTestCase` | Тестовый кейс: tool_name, test_input |
| `ValidationResult` | Финальная оценка: confidence, feedback |

---

## Manager Tools

| Инструмент | Описание |
|-----------|----------|
| `list_skills()` | Список всех навыков в реестре |
| `read_skill(name)` | Описание + промпт + код навыка |
| `create_skill(name, description, tool_code, system_prompt)` | Создать навык |
| `update_skill_tools(name, tool_code, append)` | Обновить код |
| `delete_skill(name)` | Удалить навык |
| `load_skill_tools(name)` | Загрузить @tool функции в рантайм |
| `get_skills_for_prompt()` | Описания для промптов |

---

## История диалога

Агент поддерживает многоходовые диалоги через `chat_history`:

- Формат: `[{"role": "user"|"assistant", "content": str}, ...]`
- Управляется в `main.py`, передаётся во все ноды
- Ограничение: 20 записей (10 пар user/assistant)
- Сбрасывается при команде `new`

---

## Правила генерации кода

Промпты жёстко ограничивают что может генерировать агент:

**Запрещено:**
- Заглушки: `results = []`, `pass`, `# TODO`, `# Замените`
- Плейсхолдеры: `YOUR_API_KEY`, `REPLACE_ME`, `<api_key>`, `dummy`
- API с обязательной регистрацией/ключом

**Обязательно:**
- Только бесплатные API без авторизации
- Стандартная библиотека: `urllib`, `json`, `re`, `pathlib`
- Полная обработка ошибок (`try/except`)

Рекомендуемые бесплатные API:

| Задача | API |
|--------|-----|
| Погода | `wttr.in` |
| Поиск | DuckDuckGo API |
| Геокодинг | `nominatim.openstreetmap.org` |
| Курсы валют | `open.er-api.com` |
| Wikipedia | `en.wikipedia.org/api/rest_v1/` |
| Время | `worldtimeapi.org` |

---

## Зависимости

```
langchain >= 1.2.10
langchain-openai >= 1.1.10
langgraph >= 1.0.10
langgraph-checkpoint-sqlite >= 3.0.3
omegaconf >= 2.3.0
pydantic >= 2.12.5
```

Python >= 3.13
