"""
AST-анализ кода СГЕНЕРИРОВАННЫХ навыков (реальный гейт, не строковый матчинг).

Применяется в точке записи кода (create_skill / update_skill_tools) — любой код,
который LLM пытается сохранить как навык, проходит через этот анализ. Ловит и
алиасы (`import subprocess as sp`, `from os import system as s`), и обход через
getattr по модулю. Защищённые core-навыки (subprocess для osascript и т.п.) уже
лежат на диске и через эти тулы не пишутся, поэтому политика для генерируемого
кода жёстче, чем для core.

Честные границы: статический анализ НЕ ловит динамическую сборку строк
(`__import__('o'+'s')` поймается — __import__ запрещён; а вот exec мы запрещаем
целиком). Это второй слой; первый — изолированный smoke-тест в подпроцессе
(см. utils.run_tool_sandboxed), третий — human-in-the-loop (src/hitl.py).

Обход для владельца: env AGENT_ALLOW_RISKY_SKILLS=1 отключает гейт.
"""
from __future__ import annotations

import ast

# Модули, запрещённые к импорту в генерируемых навыках целиком.
BANNED_IMPORTS = {"subprocess", "ctypes", "pty", "importlib"}

# Опасные функции верхнего уровня (вызов по имени).
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}

# Опасные атрибуты конкретных модулей: module → {attr, ...}.
BANNED_ATTRS = {
    "os": {
        "system", "popen", "execv", "execve", "execvp", "execvpe", "execl",
        "execle", "execlp", "execlpe", "spawnl", "spawnle", "spawnlp",
        "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "fork",
        "forkpty", "kill", "killpg", "setuid", "setgid",
    },
    "shutil": {"rmtree"},
}


class _Auditor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: list[str] = []
        # локальное имя → имя реального модуля (import os as o → {"o": "os"})
        self.module_aliases: dict[str, str] = {}
        # имена, импортированные из опасных модулей (from os import system as s)
        self.banned_names: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in BANNED_IMPORTS:
                self.issues.append(f"строка {node.lineno}: импорт модуля '{root}' запрещён в генерируемых навыках")
            if root in BANNED_ATTRS or root in BANNED_IMPORTS:
                self.module_aliases[alias.asname or root] = root
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in BANNED_IMPORTS:
            self.issues.append(f"строка {node.lineno}: импорт из модуля '{root}' запрещён в генерируемых навыках")
        elif root in BANNED_ATTRS:
            for alias in node.names:
                if alias.name in BANNED_ATTRS[root]:
                    self.banned_names[alias.asname or alias.name] = f"{root}.{alias.name}"
        self.generic_visit(node)

    def _module_of(self, expr: ast.expr) -> str | None:
        """Возвращает имя опасного модуля, если выражение — его алиас."""
        if isinstance(expr, ast.Name):
            return self.module_aliases.get(expr.id)
        return None

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in BANNED_CALLS:
                self.issues.append(f"строка {node.lineno}: вызов '{func.id}' запрещён")
            elif func.id in self.banned_names:
                self.issues.append(f"строка {node.lineno}: вызов '{self.banned_names[func.id]}' запрещён")
            elif func.id == "getattr" and node.args and self._module_of(node.args[0]):
                # getattr(os, 'sys'+'tem') — классический обход строкового матчинга
                self.issues.append(
                    f"строка {node.lineno}: getattr по модулю '{self._module_of(node.args[0])}' запрещён (обход анализа)"
                )
        elif isinstance(func, ast.Attribute):
            mod = self._module_of(func.value)
            if mod and func.attr in BANNED_ATTRS.get(mod, set()):
                self.issues.append(f"строка {node.lineno}: вызов '{mod}.{func.attr}' запрещён")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # ссылка на опасный атрибут без вызова (передача как callback и т.п.)
        mod = self._module_of(node.value)
        if mod and node.attr in BANNED_ATTRS.get(mod, set()):
            self.issues.append(f"строка {node.lineno}: обращение к '{mod}.{node.attr}' запрещено")
        self.generic_visit(node)


def validate_skill_code(code: str) -> tuple[bool, list[str]]:
    """
    Статический AST-гейт для генерируемого кода навыка.
    Возвращает (ok, список конкретных проблем со строками).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"SyntaxError на строке {e.lineno}: {e.msg}"]

    auditor = _Auditor()
    auditor.visit(tree)
    # дедуп с сохранением порядка
    issues = list(dict.fromkeys(auditor.issues))
    return not issues, issues


class _ModuleLevelAuditor(_Auditor):
    """Как _Auditor, но ловит опасные ВЫЗОВЫ только на уровне модуля (исполняются при импорте,
    ДО HITL). Тела функций/классов НЕ инспектирует — их код выполнится лишь при вызове тула,
    а он под HITL (+ allowlist-обёрткой у импортированных навыков). Сам импорт не баним: опасен
    не `import subprocess`, а его вызов при загрузке модуля."""

    def __init__(self) -> None:
        super().__init__()
        # алиасы модулей, которые НЕЛЬЗЯ вызывать при импорте вообще (любой их вызов на
        # уровне модуля = сайд-эффект до HITL): subprocess/ctypes/pty.
        self.danger_modules: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in BANNED_ATTRS or root in BANNED_IMPORTS:
                local = alias.asname or root
                self.module_aliases[local] = root  # только алиасы, без issue
                if root in BANNED_IMPORTS:
                    self.danger_modules.add(local)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in BANNED_ATTRS:
            for alias in node.names:
                if alias.name in BANNED_ATTRS[root]:
                    self.banned_names[alias.asname or alias.name] = f"{root}.{alias.name}"

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                and func.value.id in self.danger_modules:
            mod = self.module_aliases.get(func.value.id, func.value.id)
            self.issues.append(
                f"строка {node.lineno}: вызов '{mod}.{func.attr}' на уровне модуля "
                f"исполнится при импорте (до HITL) — запрещён"
            )
        super().visit_Call(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass  # не спускаемся в тело — оно исполнится только при вызове (под HITL)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]
    visit_ClassDef = visit_FunctionDef          # type: ignore[assignment]


def validate_module_level(code: str) -> tuple[bool, list[str]]:
    """AST-гейт уровня МОДУЛЯ: запрещает опасные сайд-эффекты при ИМПОРТЕ (exec до HITL), но
    допускает subprocess и т.п. внутри тел функций. Для импортированных навыков (OpenClaw-обёртки
    CLI), у которых своя модель защиты — HITL + runtime-allowlist бинарей."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"SyntaxError на строке {e.lineno}: {e.msg}"]

    auditor = _ModuleLevelAuditor()
    auditor.visit(tree)
    issues = list(dict.fromkeys(auditor.issues))
    return not issues, issues
