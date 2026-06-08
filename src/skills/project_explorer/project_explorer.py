import os
from typing import List, Optional
from langchain.tools import tool

@tool
def get_project_tree(root_dir: str = ".", ignore_dirs: Optional[List[str]] = None) -> str:
    """
    Возвращает строковое представление дерева проекта, начиная с root_dir.
    """
    if ignore_dirs is None:
        ignore_dirs = [".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"]
    
    tree = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(root_dir, "").count(os.sep)
        indent = " " * 4 * level
        tree.append(f"{indent}{os.path.basename(root) or root}/")
        sub_indent = " " * 4 * (level + 1)
        for f in files:
            tree.append(f"{sub_indent}{f}")
            
    return "\n".join(tree)

@tool
def read_project_file(file_path: str) -> str:
    """
    Читает содержимое файла по указанному пути.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Ошибка при чтении файла {file_path}: {str(e)}"

@tool
def find_files_by_extension(extension: str, root_dir: str = ".") -> List[str]:
    """
    Ищет все файлы с указанным расширением (например, '.py', '.md') во всем проекте.
    """
    found_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                found_files.append(os.path.join(root, file))
    return found_files
