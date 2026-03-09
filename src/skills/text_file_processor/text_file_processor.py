import os
from pathlib import Path
from langchain_core.tools import tool

@tool
def read_text_file(file_path: str) -> str:
    """Reads and returns the content of a text file.
    
    Args:
        file_path: Path to the text file to read.
    
    Returns:
        str: Content of the file or error message.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File {file_path} does not exist"
        if path.is_dir():
            return f"{file_path} is a directory, not a file"
        return path.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_text_file(file_path: str, content: str) -> str:
    """Writes text content to a file, creating it if needed.
    
    Args:
        file_path: Path to the file to write.
        content: Text content to write.
    
    Returns:
        str: Confirmation or error message.
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def search_in_files(directory: str, search_term: str, file_extension: str = '.txt') -> str:
    """Searches for text in all files with given extension in a directory.
    
    Args:
        directory: Path to directory to search.
        search_term: Text to search for.
        file_extension: File extension to filter by (default .txt).
    
    Returns:
        str: List of files containing the term or error message.
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"Directory {directory} does not exist"
        if not path.is_dir():
            return f"{directory} is not a directory"
        
        results = []
        for file in path.rglob(f"*{file_extension}"):
            try:
                content = file.read_text(encoding='utf-8')
                if search_term in content:
                    results.append(str(file))
            except:
                continue
        
        if results:
            return f"Found in:\n" + "\n".join(results)
        return f"No files contain '{search_term}'"
    except Exception as e:
        return f"Error searching files: {e}"