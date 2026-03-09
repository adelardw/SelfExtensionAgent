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
