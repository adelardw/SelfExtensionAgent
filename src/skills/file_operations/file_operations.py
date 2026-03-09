import os
from pathlib import Path
from langchain_core.tools import tool

@tool
def read_file(file_path: str) -> str:
    """Читает содержимое текстового файла и возвращает его."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Файл {file_path} не существует."
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Ошибка при чтении файла: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Записывает текст в файл. Если файл не существует, он будет создан."""
    try:
        path = Path(file_path)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Файл {file_path} успешно записан."
    except Exception as e:
        return f"Ошибка при записи файла: {e}"

@tool
def search_in_file(file_path: str, search_term: str) -> str:
    """Ищет строку в файле и возвращает все строки, содержащие её."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Файл {file_path} не существует."
        results = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if search_term in line:
                    results.append(line.strip())
        return "\n".join(results) if results else "Совпадений не найдено."
    except Exception as e:
        return f"Ошибка при поиске в файле: {e}"

@tool
def delete_file(file_path: str) -> str:
    """Удаляет файл по указанному пути."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Файл {file_path} не существует."
        os.remove(path)
        return f"Файл {file_path} успешно удалён."
    except Exception as e:
        return f"Ошибка при удалении файла: {e}"