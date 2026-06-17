"""
ASCII-арт и анимации CLI: лого SEA (перелив синий↔металлик-серый), Squid-Girl (Ika
Musume — морская тема SEA) и океан-спиннер (бегущая волна во время размышления).

Чистая косметика поверх rich. Без сети и тяжёлых зависимостей. Всё деградирует тихо
(не-TTY / нет rich) — печатается статичная синяя версия, поведение CLI не ломается.
"""
from __future__ import annotations

import time

# ── Лого SEA (ANSI-Shadow блок-стиль) ────────────────────────────────────────
SEA_LOGO = r"""███████╗███████╗ █████╗
██╔════╝██╔════╝██╔══██╗
███████╗█████╗  ███████║
╚════██║██╔══╝  ██╔══██║
███████║███████╗██║  ██║
╚══════╝╚══════╝╚═╝  ╚═╝"""

# Squid Girl (Ika Musume) теперь рендерится из РЕАЛЬНОЙ картинки в ANSI (assets_ika.ans,
# см. squid_renderable) — ручной box-арт убран (выходил «страшным»).
_SQUID_CACHE = None  # кэш Ика-renderable (баннер ре-рендерится на каждый тик анимации лого)

# Палитра перелива: синий → циан → серебро → металлик-серый → обратно (для shimmer).
_SHIMMER = ["#1e6fff", "#2a93ff", "#38c6ff", "#7fe3ff", "#bfe9f5",
            "#cbd5e1", "#9aa7b8", "#7c899c", "#9aa7b8", "#cbd5e1",
            "#bfe9f5", "#7fe3ff", "#38c6ff", "#2a93ff"]


def shimmer_color(phase: int) -> str:
    """Цвет из палитры перелива по фазе (бесконечный синий↔металлик-серый) — для статуса think."""
    return _SHIMMER[phase % len(_SHIMMER)]


# Анимированная «точечная фигурка» рядом с текстом размышления (брайль — морфится по точкам).
THINK_DOTS = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def think_dot(phase: int) -> str:
    return THINK_DOTS[phase % len(THINK_DOTS)]


# Большой набор «думательных» текстовок (gerund) — морская/общая тема, как у CLI-агентов.
THINKING_WORDS = (
    "Thinking", "Spelunking", "Waveling", "Pondering", "Brewing", "Conjuring", "Noodling",
    "Percolating", "Marinating", "Surfing", "Diving", "Fathoming", "Tidewalking", "Churning",
    "Drifting", "Navigating", "Plumbing", "Wrangling", "Cogitating", "Ruminating", "Scheming",
    "Untangling", "Foraging", "Synthesizing", "Distilling", "Voyaging", "Trawling", "Beachcombing",
    "Snorkeling", "Whirlpooling", "Mulling", "Brainstorming", "Crystallizing", "Incubating",
    "Deliberating", "Calibrating", "Wayfinding", "Tinkering", "Pathfinding", "Sleuthing",
    "Concocting", "Riffing", "Excavating", "Decoding", "Charting", "Reckoning", "Pondersurfing",
    "Bubbling", "Swirling", "Coalescing",
)


def _grad_line(line: str, phase: int, palette=_SHIMMER):
    """Строка лого с горизонтальным градиентом, сдвинутым на phase (для анимации перелива)."""
    from rich.text import Text

    t = Text()
    for i, ch in enumerate(line):
        if ch == " ":
            t.append(" ")
        else:
            t.append(ch, style=palette[(i + phase) % len(palette)])
    return t


def logo_renderable(phase: int = 0):
    """SEA-лого как rich-renderable. phase сдвигает градиент → анимация перелива в баннере
    (TUI дёргает с растущей фазой). phase=0 — статичный градиент."""
    from rich.console import Group

    lines = SEA_LOGO.splitlines()
    return Group(*[_grad_line(ln, phase + r) for r, ln in enumerate(lines)])


def shimmer_logo(console, cycles: int = 2, fps: int = 18, transient: bool = False) -> None:
    """Анимированный перелив лого SEA (синий↔металлик-серый). transient=True → после анимации
    лого ГАСНЕТ (чтобы дальше показать его статично в раскладке). Нет TTY → ничего/статика."""
    lines = SEA_LOGO.splitlines()
    try:
        from rich.console import Group
        from rich.live import Live

        if not console.is_terminal:
            raise RuntimeError("not a tty")
        frames = len(_SHIMMER) * cycles
        with Live(console=console, refresh_per_second=fps, transient=transient) as live:
            for f in range(frames):
                live.update(Group(*[_grad_line(ln, f + r) for r, ln in enumerate(lines)]))
                time.sleep(1.0 / fps)
            if not transient:
                live.update(Group(*[_grad_line(ln, r) for r, ln in enumerate(lines)]))  # статик-финал
    except Exception:  # noqa: BLE001
        if not transient:
            from rich.text import Text
            console.print(Text(SEA_LOGO, style="#38c6ff"))


def squid_renderable():
    """Squid Girl (Ika Musume) — рендер РЕАЛЬНОЙ картинки в цветные ANSI-полублоки ▀
    (из assets_ika.ans, сгенерён Pillow один раз). Файл рядом с модулем (editable/dev — живой;
    для PyInstaller добавить в datas — TODO). Нет файла → лаконичный текстовый фолбэк."""
    from pathlib import Path

    from rich.text import Text

    global _SQUID_CACHE
    if _SQUID_CACHE is not None:
        return _SQUID_CACHE
    p = Path(__file__).parent / "assets_ika.ans"
    try:
        _SQUID_CACHE = Text.from_ansi(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        _SQUID_CACHE = Text("～ Ika ～\n from the\n deep blue\n sea ♪", style="#38c6ff")
    return _SQUID_CACHE


def squid_height() -> int:
    """Высота Ика-арта в строках (для согласования высоты панели справа)."""
    from pathlib import Path

    p = Path(__file__).parent / "assets_ika.ans"
    try:
        return p.read_text(encoding="utf-8").count("\n") + 1
    except Exception:  # noqa: BLE001
        return 12


def banner_left_height() -> int:
    """Высота ЛЕВОГО блока баннера = лого + пустая строка + Ика. Панель справа берёт ту же
    высоту, чтобы блоки были вровень (не «плясали»)."""
    return len(SEA_LOGO.splitlines()) + 1 + squid_height()


def register_ocean_spinner() -> str:
    """Регистрирует спиннер 'ocean' — бегущая 🌊-волна (эмодзи) в rich. Возвращает имя
    ('ocean' при успехе, иначе 'dots'). Идемпотентно."""
    try:
        from rich import spinner as _sp

        # 🌊 «катится» по морской глади ～ туда-обратно — простая, читаемая эмодзи-волна.
        frames = ["🌊 ～ ～ ～", "～ 🌊 ～ ～", "～ ～ 🌊 ～", "～ ～ ～ 🌊",
                  "～ ～ 🌊 ～", "～ 🌊 ～ ～"]
        _sp.SPINNERS["ocean"] = {"interval": 200, "frames": frames}
        return "ocean"
    except Exception:  # noqa: BLE001
        return "dots"
