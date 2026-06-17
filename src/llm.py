"""
Единая точка для LLM-клиентов. Провайдер в config.yml: `provider` (openrouter | ollama).
Модели выбираются по РОЛИ (fast | code | embed), а не по имени — без догадок и костылей.
Фолбэк: provider=ollama → используем Ollama, если он реально генерирует, иначе OpenRouter.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()  # ключи доступны и при прямом использовании (media.py и т.п.), не только через agent

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# config.yml через резолвер: cwd-файл (project override) ИЛИ ПАКЕТНЫЙ дефолт — иначе установленный
# `sea` в чужом каталоге падал FileNotFoundError при импорте (грузил "config.yml" из cwd). Дефолтные
# модели «остаются» — они в пакетном config.yml.
from .config_paths import base_config_path
_cfg = OmegaConf.load(str(base_config_path()))
_active: str | None = None  # кэш активного провайдера за сессию
_override: dict = {"provider": None, "model": None}  # рантайм-выбор из CLI (/model)


def set_provider(provider_name: str | None, model: str | None = None) -> None:
    """Рантайм-переключение провайдера/модели (CLI /model). Сбрасывает кэш health-check."""
    global _active
    _override["provider"] = provider_name
    _override["model"] = model
    _active = None


def _cli_override(key: str, default=None):
    """Пользовательская настройка из config.local.yml (cli.<key>) — заполняется из /config.
    Ленивый импорт, чтобы избежать цикла; битый/отсутствующий конфиг не ломает запуск."""
    try:
        from .cli_config import get_cli
        return get_cli(key, default)
    except Exception:  # noqa: BLE001
        return default


def api_key() -> str | None:
    """Ключ: env (приоритет) → пользовательский ввод из настроек (config.local.yml)."""
    return (os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            or _cli_override("api_key") or None)


def api_key_source() -> str:
    """Откуда взят ключ — для отображения в настройках (без раскрытия самого ключа)."""
    if os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return "env"
    if _cli_override("api_key"):
        return "настройки"
    return "не задан"


def validate_credentials(key: str | None = None, base_url: str | None = None,
                         model: str | None = None) -> tuple[bool, str]:
    """Живая проверка ключа/endpoint: минимальный chat-запрос (max_tokens=1) к
    OpenAI-совместимому base_url. Работает для openrouter и любого кастомного endpoint."""
    key = key or api_key()
    base = (base_url or _cli_override("base_url") or OPENROUTER_BASE).rstrip("/")
    if not key:
        return False, "API-ключ не задан"
    mdl = model or model_for("fast")
    body = json.dumps({"model": mdl, "messages": [{"role": "user", "content": "ping"}],
                       "max_tokens": 1}).encode()
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            json.loads(r.read())
        return True, "ключ валиден ✓"
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        if e.code in (401, 403):
            return False, "ключ отклонён (401/403 — неверный ключ или нет доступа)"
        return False, f"endpoint ответил HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return False, f"не удалось проверить ({type(e).__name__}): {str(e)[:60]}"


def _ollama_works() -> bool:
    """Ollama не просто отвечает, а РЕАЛЬНО генерирует (ловит сломанный движок)."""
    oll = _cfg.get("ollama", {})
    base = oll.get("base_url", "http://localhost:11434/v1").rstrip("/").removesuffix("/v1")
    model = oll.get("model", "llama3.1")
    try:
        req = urllib.request.Request(
            base + "/api/generate",
            data=json.dumps({"model": model, "prompt": "hi", "stream": False,
                             "options": {"num_predict": 1}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            json.loads(r.read())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[llm] ⚠ Ollama не генерирует ({str(e)[:80]}) → fallback на OpenRouter.")
        return False


def provider() -> str:
    global _active
    if _active is not None:
        return _active
    p = _override["provider"] or _cfg.get("provider", "openrouter")
    if p == "ollama" and not _ollama_works():
        p = "openrouter"
    _active = p
    return p


def model_for(role: str) -> str:
    """Имя модели для роли при активном провайдере. role: fast | code | deep | embed.
    'deep' — редкие тяжёлые вызовы (heavy-ревью); фолбэк на code-модель."""
    if provider() == "ollama":
        oll = _cfg.get("ollama", {})
        if _override["model"] and role in ("fast", "code", "deep"):
            return _override["model"]  # модель, выбранная из CLI
        fast = oll.get("model", "llama3.1")
        return {"fast": fast, "code": oll.get("code_model") or fast,
                "deep": oll.get("deep_model") or oll.get("code_model") or fast,
                "embed": oll.get("embed_model", "nomic-embed-text")}.get(role, fast)
    # openrouter/совместимый: имена моделей по ролям можно задать из настроек (cli.*),
    # иначе берём из config.yml.
    if role == "code":
        return _cli_override("code_model") or _cfg.get("code_model", {}).get("name", "gpt-4o-mini")
    if role == "deep":
        return (_cli_override("deep_model") or (_cfg.get("deep_model", {}) or {}).get("name")
                or model_for("code"))
    if role == "embed":
        return _cli_override("embed_model") or _cfg.get("memory", {}).get("embedding_model", "openai/text-embedding-3-small")
    return _cli_override("model") or _cfg.get("model", {}).get("name", "gpt-4o-mini")


def _base_and_key() -> tuple[str, str]:
    if provider() == "ollama":
        return _cfg.get("ollama", {}).get("base_url", "http://localhost:11434/v1"), "ollama"
    # openrouter / совместимый: endpoint можно переопределить из настроек (cli.base_url)
    return (_cli_override("base_url") or OPENROUTER_BASE), (api_key() or "")


def chat(role: str = "fast", temperature: float = 0.0):
    """ChatOpenAI для активного провайдера + модель по роли (fast|code|deep).
    К каждому клиенту привязан run-budget callback — все вызовы (включая под-агентов)
    учитываются в токен-бюджете прогона (см. runbudget)."""
    from langchain_openai.chat_models import ChatOpenAI

    from .runbudget import callback

    base, key = _base_and_key()
    # max_tokens: потолок ВЫХОДА одного вызова. Без него thinking-модель (gemini-2.5-flash-lite)
    # тратит output на размышление и ОБРЕЗАЕТ structured-JSON (decompose) → LengthFinishReasonError
    # → деградация «один шаг = весь запрос» (живой баг на GAIA L2). Щедрый кап даёт JSON дозреть.
    max_out = int(_cfg.get("agent", {}).get("max_output_tokens", 8000))
    # ЖЁСТКИЙ ТАЙМАУТ + мало ретраев: зависший API-вызов не должен морозить агента на
    # дефолтные 600с × ретраи (это плодило «зомби на 0% CPU» в eval и фризы в проде).
    # 60с: реальный fast/code-вызов <30с; 60с = явный хэнг. Внутренний дедлайн прогона
    # (MAX_RUN_SECONDS=150) тогда успевает прерваться до враппера eval (240с): +1 вызов ≤60с
    # = ≤210с (живой баг: TimeoutError на 240с, внутр. стоп не срабатывал при timeout=90).
    llm_timeout = float(_cfg.get("agent", {}).get("llm_timeout", 60))
    return ChatOpenAI(api_key=key or "x", base_url=base, model=model_for(role),
                      temperature=temperature, callbacks=[callback()],
                      timeout=llm_timeout, max_retries=1, max_tokens=max_out)


def active_summary() -> str:
    """Строка для баннера — РЕАЛЬНО активные провайдер и модели."""
    p = provider()
    fast, code = model_for("fast"), model_for("code")
    return f"{p} · {fast}" + (f" · код: {code}" if code != fast else "")
