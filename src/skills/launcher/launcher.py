import os
import subprocess
from langchain_core.tools import tool

@tool
def launch_app(app_name: str) -> str:
    """Launch a macOS application by name.

    Args:
        app_name: Name of the app (e.g., 'Telegram', 'FaceTime').
    """
    # Бенч/eval/безопасный режим: не запускаем реальное приложение (крадёт фокус у юзера).
    if os.getenv("AGENT_DRY_RUN") or os.getenv("AGENT_EVAL_MODE") == "1":
        return f"[dry-run] запуск {app_name}"
    try:
        subprocess.run(["open", "-a", app_name], check=True)
        return f"Приложение {app_name} успешно запущено."
    except Exception as e:
        return f"Ошибка при запуске {app_name}: {e}"
