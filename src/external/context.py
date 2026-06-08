"""
Контекст внешнего агента (A2A / MCP) в состоянии графа.

Когда наш агент взаимодействует с чужим агентом (Agent-to-Agent) или с MCP-сервером,
важно держать в состоянии, С КЕМ мы говорим и что он умеет — иначе декомпозиция и
исполнение «не видят» внешние возможности и ограничения.

Здесь — модель этого контекста + process-local реестр по user_id. Реальная привязка
к MCP-клиенту/A2A-каналу подключается позже: интеграция вызывает set_external_context()
при установлении связи (например, после загрузки MCP-сервера из поиска), а граф просто
читает get_external_context() в recall и прокидывает в промпты.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ExternalAgentContext(BaseModel):
    name: str = Field(default="", description="Имя внешнего агента / MCP-сервера")
    source: str = Field(default="", description="Тип канала: 'a2a' | 'mcp' | ''")
    capabilities: list[str] = Field(default_factory=list, description="Что внешний агент умеет (tools/skills)")
    notes: str = Field(default="", description="Договорённости, ограничения, формат обмена")

    def is_present(self) -> bool:
        return bool(self.name)


# user_id -> активный внешний контекст (один на пользователя; расширяемо до списка).
_REGISTRY: dict[str, ExternalAgentContext] = {}


def set_external_context(user_id: str, ctx: ExternalAgentContext) -> None:
    _REGISTRY[user_id] = ctx


def clear_external_context(user_id: str) -> None:
    _REGISTRY.pop(user_id, None)


def get_external_context(user_id: str) -> ExternalAgentContext:
    return _REGISTRY.get(user_id, ExternalAgentContext())


def format_external_context(ctx: dict | None) -> str:
    """Готовый текст для инъекции в промпты."""
    if not ctx or not ctx.get("name"):
        return "Внешних агентов/MCP в этой задаче нет."
    caps = ", ".join(ctx.get("capabilities", [])) or "(не указаны)"
    parts = [f"Взаимодействие с внешним агентом «{ctx['name']}» (канал: {ctx.get('source') or '?'})."]
    parts.append(f"Возможности: {caps}.")
    if ctx.get("notes"):
        parts.append(f"Заметки: {ctx['notes']}")
    return " ".join(parts)
