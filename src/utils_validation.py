import ast
import sys
import subprocess
from pathlib import Path
from src.core_config import logger

def validate_python_code(code: str) -> tuple[bool, str]:
    """
    Проверяет код на синтаксические ошибки и базовую безопасность.
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    # Запрещенные паттерны (базовая безопасность)
    forbidden = ["os.system", "subprocess.Popen", "eval(", "exec("]
    for item in forbidden:
        if item in code:
            return False, f"Security risk: use of {item} is restricted."
            
    return True, "OK"

def run_ruff_check(file_path: str) -> bool:
    """
    Запускает ruff для проверки качества кода.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", file_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.warning("ruff_issues", issues=result.stdout)
            return False
        return True
    except Exception as e:
        logger.error("ruff_failed", error=str(e))
        return False
