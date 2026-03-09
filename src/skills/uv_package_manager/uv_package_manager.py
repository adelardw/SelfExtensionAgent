import subprocess
import sys
from langchain_core.tools import tool

@tool
def uv_package_manager(package_name: str, version: str = "") -> str:
    """Устанавливает Python-пакет через uv."""
    try:
        # Проверяем, установлен ли uv
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return "Ошибка: uv не установлен. Установите его через 'pip install uv'."

    # Формируем команду для установки
    package_spec = f"{package_name}{version}" if version else package_name
    command = ["uv", "pip", "install", package_spec]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return f"Пакет {package_spec} успешно установлен.\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Ошибка установки пакета {package_spec}:\n{e.stderr}"