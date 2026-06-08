from pydantic import BaseModel, Field
from typing import Literal


class RouteDecision(BaseModel):
    """Выход роутера — определяет какую ветку графа использовать."""
    reasoning: str = Field(description="Краткое обоснование выбранного маршрута")
    route: Literal["create_skill", "use_skills"] = Field(
        description="'create_skill' — ни один существующий навык не подходит, "
                    "'use_skills' — существующие навыки покрывают задачу"
    )


class SkillSelection(BaseModel):
    """Выход селектора навыков — выбирает подходящие навыки из реестра."""
    reasoning: str = Field(description="Почему выбраны именно эти навыки")
    selected_skills: list[str] = Field(
        description="Точные snake_case имена навыков из реестра"
    )


class ExecutionPlan(BaseModel):
    """Выход планировщика — упорядоченные шаги выполнения."""
    reasoning: str = Field(description="Общий подход к решению задачи")
    steps: list[str] = Field(description="Конкретные, упорядоченные шаги выполнения")


class NodeGradient(BaseModel):
    """Текстовый «градиент» одной ноды: как ИМЕННО ей стоит измениться."""
    node: str = Field(description="Имя ноды графа (из списка оптимизируемых)")
    critique: str = Field(description="Конкретно что нода делает не так и как улучшить (её локальный градиент)")


class NodeGradients(BaseModel):
    """Backward по графу: распределённые по нодам текстовые градиенты из батча неудач."""
    gradients: list[NodeGradient] = Field(
        description="Только для нод, реально виноватых в неудачах. Не вали вину на ноды, "
                    "которые отработали корректно. Пусто — если винить нечего.",
        default_factory=list,
    )


class CognitiveAssessment(BaseModel):
    """
    Единый «акт мышления»: за ОДИН вызов и понять цель, и выбрать режим.
    Слияние goal+reflexion ради бюджета (2 LLM-вызова → 1) — и это когнитивно
    честнее: человек оценивает суть задачи и «как думать» одновременно.
    """
    # — цель —
    aim: str = Field(description="Краткая формулировка намерения текущего запроса")
    is_standing: bool = Field(description="Это многошаговая/долгая цель, которую держать в контексте?")
    completes_active: bool = Field(description="Запрос завершает активную стоящую цель?", default=False)
    standing_goal: str = Field(description="Текст стоящей цели для удержания (если is_standing), иначе ''", default="")
    success_criteria: list[str] = Field(
        description="Rubric цели: проверяемый чек-лист (для is_standing), иначе пусто", default_factory=list
    )
    # — режим мышления —
    mode: Literal["fast", "deliberate", "clarify"] = Field(
        description="'fast' — быстрый интуитивный ответ без инструментов; "
                    "'deliberate' — обдумывание с инструментами/многошаговостью/свежими данными; "
                    "'clarify' — переспросить при неоднозначности. Бюджет важен: не уходи в deliberate зря."
    )
    needs_tools: bool = Field(description="Нужны ли внешние инструменты/навыки")


class ReflexionDecision(BaseModel):
    """
    Self-Reflexion Choice: выбор режима мышления (модель System 1 / System 2).
    Главный рычаг бюджета — не гонять дорогой путь там, где хватит быстрого.
    """
    mode: Literal["fast", "reason", "deliberate", "clarify"] = Field(
        description="Режим/тип мышления по анализу задачи: "
                    "'fast' — быстрый интуитивный ответ без инструментов (приветствия, простое, известное); "
                    "'reason' — глубокое пошаговое рассуждение БЕЗ инструментов (сложные вопросы, "
                    "анализ, математика, логика — где думать надо много, но внешние данные не нужны); "
                    "'deliberate' — обдумывание с инструментами/навыками, многошаговостью, свежими данными; "
                    "'clarify' — задача неоднозначна, дешевле переспросить."
    )
    needs_tools: bool = Field(description="Нужны ли внешние инструменты/навыки для ответа")
    rationale: str = Field(description="Кратко почему выбран этот режим")


class SubTask(BaseModel):
    """Один пункт плана (атомарный подшаг декомпозиции)."""
    goal: str = Field(description="Что конкретно сделать на этом шаге — одно атомарное действие")
    done_check: str = Field(description="Проверяемый критерий, что шаг выполнен (мини-rubric шага)")


class TaskDecomposition(BaseModel):
    """
    Смешанный формат: связный ризонинг, ВНУТРИ которого рождается план.
    reasoning — рассуждение о подходе; subtasks — извлечённые из него пункты.
    """
    reasoning: str = Field(description="Связное рассуждение о подходе, по ходу которого формируется план")
    subtasks: list[SubTask] = Field(description="Упорядоченные атомарные подшаги (обычно 1–6)")


class StepOutcome(BaseModel):
    """По-пунктовая валидация: пройден ли подшаг по своему done_check."""
    passed: bool = Field(description="Подшаг выполнен и удовлетворяет done_check?")
    note: str = Field(description="Что получилось / в чём проблема (для ретрая или контекста след. шага)")


class SGRCreateResult(BaseModel):
    """Результат ревью созданного навыка (SGR ветки создания)."""
    is_valid: bool = Field(description="Навык корректный, полный и готов к использованию?")
    confidence: float = Field(description="Уверенность 0.0–1.0", ge=0.0, le=1.0)
    issues: list[str] = Field(description="Найденные проблемы (пусто если валидный)", default_factory=list)
    suggestion: str = Field(description="Как исправить проблемы (пусто если валидный)", default="")


class SkillTestCase(BaseModel):
    """Тестовый кейс для smoke-теста навыка."""
    tool_name: str = Field(
        description="Имя @tool функции для тестирования (то, что передано в декоратор @tool или имя функции)"
    )
    test_input: dict = Field(
        description="Аргументы для вызова: {имя_параметра: тестовое_значение}. "
                    "Используй простые, быстрые, реалистичные значения."
    )
    expected_behavior: str = Field(
        description="Краткое описание ожидаемого поведения (для логирования)"
    )


class ValidationResult(BaseModel):
    """Финальная валидация ответа агента (SGR конца графа)."""
    is_valid: bool = Field(description="Ответ полностью соответствует запросу пользователя?")
    confidence: float = Field(description="Уверенность 0.0–1.0", ge=0.0, le=1.0)
    feedback: str = Field(description="Краткая оценка качества ответа")


class UserFact(BaseModel):
    """Один устойчивый факт о пользователе (семантическая память / персонализация)."""
    key: str = Field(description="Короткий стабильный ключ в snake_case, напр. 'язык_общения', 'часовой_пояс', 'предпочитаемый_стек'")
    value: str = Field(description="Значение факта одной фразой")
    importance: float = Field(description="Насколько это важно для долгой памяти, 0.0–1.0", ge=0.0, le=1.0, default=0.5)
    tags: list[str] = Field(
        description="1–3 тега-темы для группировки и связного поиска (напр. 'работа', 'личное', 'стек', 'предпочтения')",
        default_factory=list,
    )


class MemoryExtraction(BaseModel):
    """Извлечение устойчивых фактов о пользователе из последнего обмена."""
    facts: list[UserFact] = Field(
        description="ТОЛЬКО устойчивые, переиспользуемые факты/предпочтения о пользователе. "
                    "НЕ сохраняй разовые детали задачи, погоду, временные данные. "
                    "Пустой список — если ничего достойного долгой памяти нет.",
        default_factory=list,
    )


class ReflectionResult(BaseModel):
    """Вывод высокого порядка, синтезированный из недавних эпизодов."""
    insights: list[str] = Field(
        description="1–3 обобщающих вывода о паттернах пользователя или о том, "
                    "как лучше ему помогать. Каждый — самодостаточная фраза.",
        default_factory=list,
    )


class GoalAssessment(BaseModel):
    """Само-рефлексия: определяет цель запроса и нужно ли держать её в контексте."""
    aim: str = Field(description="Краткая формулировка цели/намерения ТЕКУЩЕГО запроса (1 фраза)")
    is_standing: bool = Field(
        description="True, если это многошаговая/долгая цель, которую нужно ДЕРЖАТЬ В КОНТЕКСТЕ "
                    "между сообщениями (проект, серия связанных задач). False — для разовых вопросов."
    )
    completes_active: bool = Field(
        description="True, если этот запрос ЗАВЕРШАЕТ ранее установленную активную стоящую цель.",
        default=False,
    )
    standing_goal: str = Field(
        description="Если is_standing — текст цели для удержания в контексте (можно уточнить/обновить "
                    "существующую). Иначе пустая строка.",
        default="",
    )
    success_criteria: list[str] = Field(
        description="Rubric: проверяемый чек-лист критериев «когда цель выполнена» (1–5 пунктов, "
                    "каждый — отдельное измеримое условие). Заполняй для is_standing; иначе пусто. "
                    "Эти критерии будут использоваться валидатором как rubric для грейдинга.",
        default_factory=list,
    )
