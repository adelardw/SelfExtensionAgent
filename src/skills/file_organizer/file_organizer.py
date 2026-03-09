import os
import shutil
from pathlib import Path
from langchain_core.tools import tool

@tool
def file_organizer(directory_path: str) -> str:
    """Organizes files in a directory by moving them into subdirectories based on their file extensions.

    Args:
        directory_path: The path to the directory to organize.

    Returns:
        str: A confirmation message listing the number of files moved and their new locations.
    """
    try:
        directory = Path(directory_path)
        if not directory.is_dir():
            return f"Error: {directory_path} is not a valid directory."

        files_moved = 0
        result_message = []

        for file_path in directory.iterdir():
            if file_path.is_file():
                extension = file_path.suffix.lower()
                if extension:
                    extension_dir = directory / extension[1:]  # Remove the dot
                    extension_dir.mkdir(exist_ok=True)
                    new_path = extension_dir / file_path.name
                    shutil.move(str(file_path), str(new_path))
                    files_moved += 1
                    result_message.append(f"Moved {file_path.name} to {extension_dir.name}/")

        if files_moved == 0:
            return "No files were moved. The directory may already be organized or empty."
        else:
            return f"Successfully moved {files_moved} files:\n" + "\n".join(result_message)
    except Exception as e:
        return f"Error organizing files: {e}"