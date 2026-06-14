"""
Импорт ВНЕШНЕГО навыка (папка или .zip от пользователя) с проверкой ПЕРЕД приёмом.

Пайплайн (недоверенный код НЕ исполняется в процессе агента):
  1. КАРАНТИН — zip распаковывается / папка копируется во временный каталог
     (анти-zip-slip: члены архива не должны выходить за его пределы);
  2. AST-ГЕЙТ — статический разбор кода (`validate_skill_code`), без запуска;
  3. SMOKE — импорт модуля в ИЗОЛИРОВАННОМ subprocess (rlimits + опц. syscall-sandbox),
     проверяем, что грузится и содержит ≥1 @tool-функцию;
  4. ВЕРДИКТ — только при успехе всех этапов навык копируется в src/skills/<name>/
     и регистрируется в registry.json. Иначе — отказ с причиной, ничего не ставится.

Формат навыка: каталог с `<name>.py` (функции @tool) и опционально `<name>.md` (инструкция).

CLI:  python -m src.skills_import <путь_к_папке_или_zip>
"""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from .tools.skill_creation import SKILLS_DIR, _is_protected, _load_registry, _save_registry
from .utils import run_python_sandboxed
from .utils_validation import validate_skill_code


def _quarantine(src: Path) -> tuple[Path | None, Path | None, str]:
    """Распаковать/скопировать в карантин. Возвращает (skill_dir, quarantine_root, err)."""
    q = Path(tempfile.mkdtemp(prefix="skill_quarantine_"))
    if src.is_file() and src.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(src) as z:
                qr = q.resolve()
                for m in z.namelist():
                    target = (q / m).resolve()
                    if target != qr and not str(target).startswith(str(qr) + "/"):
                        return None, q, f"небезопасный путь в zip (zip-slip): {m}"
                z.extractall(q)
        except zipfile.BadZipFile:
            return None, q, "битый zip-архив"
    elif src.is_dir():
        shutil.copytree(src, q / src.name)
    else:
        return None, q, "путь — не папка и не .zip"

    pys = [p for p in q.rglob("*.py") if "__pycache__" not in str(p)]
    if not pys:
        return None, q, "в навыке нет ни одного .py файла"
    # конвенция: <name>/<name>.py; иначе берём первый .py
    cand = next((p for p in pys if p.parent.name == p.stem), pys[0])
    return cand.parent, q, ""


def import_skill(src_path: str) -> tuple[bool, str]:
    """Главный пайплайн импорта. Возвращает (accepted, verdict_text)."""
    src = Path(src_path).expanduser()
    if not src.exists():
        return False, f"❌ не найден путь: {src}"

    skill_dir, q_root, err = _quarantine(src)
    try:
        if err or skill_dir is None:
            return False, f"❌ карантин: {err}"

        py = next((p for p in skill_dir.glob("*.py") if "__pycache__" not in str(p)), None)
        if py is None:
            return False, "❌ нет .py в каталоге навыка"
        name = py.stem
        if not name.isidentifier() or name.startswith("_"):
            return False, f"❌ недопустимое имя навыка: '{name}'"
        if _is_protected(name):
            return False, f"❌ '{name}' — защищённый core-навык, перезапись запрещена"

        code = py.read_text(encoding="utf-8", errors="replace")

        # 1) AST-ГЕЙТ (статически, без исполнения)
        ok, issues = validate_skill_code(code)
        if not ok:
            return False, "❌ AST-гейт отклонил код:\n  • " + "\n  • ".join(issues)

        # 2) SMOKE-ИМПОРТ в ПЕСОЧНИЦЕ (отдельный процесс, rlimits) — битый импорт / нет @tool
        probe = (
            "import importlib.util, json\n"
            f"spec = importlib.util.spec_from_file_location('imported_skill', {str(py)!r})\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            "tools = [getattr(m,a).name for a in dir(m) "
            "if hasattr(getattr(m,a),'name') and hasattr(getattr(m,a),'invoke')]\n"
            "print('TOOLS=' + json.dumps(tools))\n"
        )
        sok, out = run_python_sandboxed(probe, timeout=25)
        if not sok:
            return False, f"❌ smoke-импорт в песочнице упал: {out}"
        tools: list[str] = []
        for ln in out.splitlines():
            if ln.startswith("TOOLS="):
                tools = json.loads(ln[len("TOOLS="):])
        if not tools:
            return False, ("❌ навык не содержит ни одной @tool-функции "
                           "(нужен `from langchain_core.tools import tool`)")

        # 3) ВЕРДИКТ ✅ → установка + регистрация
        dest = SKILLS_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(skill_dir, dest, ignore=shutil.ignore_patterns("__pycache__"))
        md = dest / f"{name}.md"
        desc = md.read_text(encoding="utf-8", errors="replace")[:400] if md.exists() else ""
        reg = _load_registry()
        now = datetime.now().isoformat()
        reg[name] = {
            "created_at": reg.get(name, {}).get("created_at", now), "updated_at": now,
            "version": reg.get(name, {}).get("version", 0) + 1,
            "description": desc or f"Imported skill {name}",
            "has_tools": True, "has_system_prompt": md.exists(), "imported": True,
        }
        _save_registry(reg)
        return True, (f"✅ навык '{name}' ПРИНЯТ (AST ✓ · smoke ✓): тулы [{', '.join(tools)}]. "
                      f"Зарегистрирован в registry.json → доступен в следующем прогоне.")
    finally:
        if q_root and q_root.exists():
            shutil.rmtree(q_root, ignore_errors=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("использование: python -m src.skills_import <путь_к_папке_или_zip>")
        raise SystemExit(2)
    accepted, verdict = import_skill(sys.argv[1])
    print(verdict)
    raise SystemExit(0 if accepted else 1)
