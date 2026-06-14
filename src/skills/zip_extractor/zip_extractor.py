import zipfile
import io
import os
from pathlib import Path
from langchain_core.tools import tool

@tool
def extract_and_read_zip(zip_path: str, file_inside: str = "") -> str:
    """Извлекает содержимое ZIP-архива. Если указан file_inside, читает только его.
    
    Args:
        zip_path: Путь к ZIP-файлу.
        file_inside: Имя файла внутри архива для чтения (если пусто, показывает список файлов).
    
    Returns:
        str: Содержимое файла или список файлов в архиве.
    """
    try:
        path = Path(zip_path)
        if not path.exists():
            return f"Файл {zip_path} не найден."
        
        with zipfile.ZipFile(path, 'r') as zf:
            if file_inside:
                if file_inside not in zf.namelist():
                    return f"Файл {file_inside} не найден в архиве. Доступны: {', '.join(zf.namelist())}"
                with zf.open(file_inside) as f:
                    content = f.read().decode('utf-8', errors='replace')
                    return content
            else:
                return f"Файлы в архиве:\n" + "\n".join(zf.namelist())
    except Exception as e:
        return f"Ошибка: {e}"