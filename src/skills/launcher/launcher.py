import subprocess
from langchain_core.tools import tool

@tool
def launch_app(app_name: str) -> str:
    """Launch a macOS application by name.
    
    Args:
        app_name: Name of the app (e.g., 'Telegram', 'FaceTime').
    """
    try:
        subprocess.run(["open", "-a", app_name], check=True)
        return f"Приложение {app_name} успешно запущено."
    except Exception as e:
        return f"Ошибка при запуске {app_name}: {e}"
