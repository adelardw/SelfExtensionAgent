# Skill: phone_control (core)

Структурное управление Android-телефоном через **ADB** (не скриншоты).
`uiautomator dump` → XML-дерево экрана; тап по элементу (центр bounds), ввод текста,
системные клавиши. Тот же принцип, что AX на десктопе, но для телефона.

## Инструменты
- `phone_ui()` — структурное дерево текущего экрана (текст/id/clickable/координаты).
- `phone_tap(query)` — тап по элементу с текстом/id, содержащим query.
- `phone_type(text)` — ввод текста в активное поле.
- `phone_key(key)` — back/home/enter/tab/menu/power.
- `phone_open_app(package)` — запуск приложения по пакету.
- `phone_apps(filter)` — список установленных пакетов.

## Важно
- Нужен `adb` (brew install android-platform-tools) + устройство с USB-отладкой.
- AGENT_DRY_RUN=1 — действия не выполняются (чтение/список — безопасны).
