"""
Full-screen TUI (Textual) — ДЕФОЛТНЫЙ интерфейс `sea`.

Хедер-баннер [SEA-лого + Ика + гайд] (рефлоит на resize), прокручиваемая лента диалога,
рамка ввода снизу (нативная, всегда закрыта, resize). Слэш-команды + HITL/clarify-модалки —
полный паритет с прежним REPL. Старый line-mode REPL остаётся как `sea --repl` (страховка).

Интеграция: тот же граф (src.agent.build_graph), HITL через set_confirmer/set_clarifier →
Textual-модалки (push_screen_wait), память/чаты/настройки — те же src-модули.
"""
from __future__ import annotations

import uuid

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Input, Label, RichLog, Static

# Слэш-команды — для ghost-подсказок (автодополнение при вводе «/») в поле ввода.
_COMMANDS = ["/help", "/clear", "/new", "/init", "/auto", "/model", "/chats", "/fav", "/facts",
             "/goal", "/usage", "/traces", "/diagnose", "/improve", "/attach", "/voice", "/kb",
             "/compact", "/sync", "/token", "/key", "/provider", "/config", "exit"]


def _banner(phase: int = 0) -> Table:
    """Renderable баннера: слева [SEA-лого + Ика], справа панель-гайд той же высоты (expand=True
    → тянется на ширину; Textual ре-рендерит на resize → рефлоу). phase — фаза перелива лого."""
    from src import cli_art

    try:
        from src import hitl
        from src.llm import active_summary
        active, work = active_summary(), hitl.work_mode()
    except Exception:  # noqa: BLE001 — баннер не должен падать до прогрева
        active, work = "—", "—"
    lines = [
        f"[bold]Active:[/] [#38c6ff]{active}[/]",
        f"[bold]Work mode:[/] {work}  [dim](/auto to switch)[/]",
        "[bold]Thinking modes:[/] ⚡FAST 🧠REASON 🤚ACT 🛠DELIBERATE 🏗HEAVY ❓CLARIFY",
        "[bold]Physical web:[/] [dim]just say it — “play music <A>”, “open my favorites”, “pause”, "
        "“find the movie <X>”, “order <food>” — the agent picks the service itself[/]",
        "[bold]Commands:[/] [dim]/help /model /auto /chats /new /init /facts /goal /diagnose /traces "
        "/improve /usage  ·  exit[/]",
        "[bold]Chats:[/] [dim]/chats — history · /fav — ★favorite · /compact (=/compress) → "
        "COMPACT.md · /sync — rebuild SEA.md[/]",
        "[bold]Files:[/] [dim]/attach <file> — into the session · /voice — voice 🎙 · /kb add|ls|find — "
        "personal knowledge base (graph)[/]",
    ]
    # ПРИВЕТСТВЕННЫЙ баннер (ВРЕМЕННЫЙ — исчезает после первой отправки): лого SEA + Ика слева,
    # панель-инструкция справа той же высоты. Ика — ТОЛЬКО здесь (в приветствии), НИКОГДА в ленте чата.
    h = cli_art.banner_left_height()
    inner = max(len(lines), h - 2)  # текст заполняет высоту панели (вровень с лого+Икой слева)
    gaps = max(1, len(lines) - 1)
    blank = max(0, inner - len(lines))
    spread: list[str] = []
    for i, ln in enumerate(lines):
        spread.append(ln)
        if i < len(lines) - 1:
            nb = blank // gaps + (1 if i < blank % gaps else 0)
            spread.extend([""] * nb)
    body = Text.from_markup("\n".join(spread))
    bcolor = "#3b82f6" if (phase // 4) % 2 == 0 else "#94a3b8"  # перелив рамки синий↔металлик
    panel = Panel(body, title="🌊 Self-Extension Agent", border_style=bcolor, expand=True, height=h)
    left = Group(cli_art.logo_renderable(phase), "", cli_art.squid_renderable())
    grid = Table.grid(padding=(0, 4), expand=True)
    grid.add_column()
    grid.add_column(ratio=1)
    grid.add_row(left, panel)
    return grid


# Палитра модалок (как в Claude Code): акцент-фокус, зелёный-выбор, dim-подсказка.
_M_FOCUS = "#88c0d0"   # бирюзовый — указатель ❯ и подпись сфокусированной строки
_M_PICK = "#a3be8c"    # зелёный — отметка [✓] и подпись выбранной строки
_M_DIM = "#5b6472"     # приглушённый — подсказки/why/футер
_M_OTHER = "__OTHER__"
_M_SUBMIT = "__SUBMIT__"  # строка «✓ Done» в мультиселекте: Enter на ней — отправить


class _ChoiceModal(ModalScreen[str]):
    """База для clarify/HITL в стиле Claude Code: компактная панель ВНИЗУ экрана (не окно
    перед лицом), указатель ❯ у строки в фокусе, отметка [✓] зелёная (пусто [ ] — не выбрано,
    НИКАКИХ крестов), «✎ свой вариант» прямо в списке (доступен стрелками, без Tab), печать
    идёт в свой вариант, dim-подсказка снизу. Полностью с клавиатуры, без кнопок.

    Рендер — один Static, перерисовывается на каждое нажатие. Подкласс задаёт заголовок,
    список строк (rows: list[(label, value)]), мультиселект и сбор ответа."""

    CSS = """
    #cm { dock: bottom; width: 100%; height: auto; padding: 1 2; background: $surface;
          border-top: solid #3b4252; }
    """

    # Навигация/выбор — через ИМЕННОВАННЫЕ BINDINGS экрана (срабатывают даже без фокусируемого
    # виджета — в реальном терминале on_key у экрана без фокуса мог НЕ ловиться → мультиселект «не
    # работал»). Свободный текст и цифры-хоткеи добираются через on_key (печатные символы).
    BINDINGS = [
        Binding("up", "nav_up", show=False),
        Binding("down", "nav_down", show=False),
        Binding("space", "toggle", show=False),
        Binding("enter", "submit", show=False),
        Binding("escape", "skip", show=False),
        Binding("tab", "nav_down", show=False),
        Binding("shift+tab", "nav_up", show=False),
    ]

    multi = True  # мультиселект (clarify); ConfirmModal переопределит на False

    def __init__(self, head: str, rows: list[tuple[str, str]], why: str = "", chip: str = "") -> None:
        super().__init__()
        self._head, self._rows, self._why, self._chip = head, rows, why, chip
        self._focus = 0
        self._picked: set[int] = set()
        self._other = ""

    def compose(self) -> ComposeResult:
        s = Static(Text(""), id="cm")
        s.can_focus = True  # БЕЗ фокусируемого виджета фокус остаётся на фоновом #prompt → клавиши
        yield s             # уходят туда, а не в модалку (стрелки «не двигали»). Фокусируем в on_mount.

    def on_mount(self) -> None:
        self._redraw()
        self.query_one("#cm", Static).focus()

    # ── рендер строки-за-строкой (rich.Text), как ListItem в Claude Code ──────────
    def _redraw(self) -> None:
        t = Text()
        if self._chip:
            t.append(self._chip + "  ", style=f"bold {_M_FOCUS}")
        t.append(self._head + "\n", style="bold")
        if self._why:
            t.append(self._why + "\n", style=_M_DIM)
        t.append("\n")
        for i, (label, value) in enumerate(self._rows):
            focused = i == self._focus
            picked = i in self._picked
            line = Text()
            line.append("❯ " if focused else "  ", style=_M_FOCUS if focused else "")
            if value == _M_SUBMIT:  # строка-кнопка «Done» (без коробочки)
                line.append("✓ ", style=_M_PICK)
                line.append(label, style=f"bold {_M_FOCUS}" if focused else f"bold {_M_PICK}")
            elif value == _M_OTHER:  # свой вариант — без коробочки, поле ввода
                shown = self._other if self._other else "type your own option…"
                line.append("✎ ", style=_M_FOCUS if focused else _M_DIM)
                line.append(shown + ("▏" if focused else ""),
                            style=(_M_FOCUS if focused else _M_DIM) if not self._other else "")
            else:  # обычный вариант — коробочка [✓]/[ ] в мультиселекте
                if self.multi:
                    line.append("[✓] " if picked else "[ ] ", style=_M_PICK if picked else _M_DIM)
                style = _M_PICK if picked else (_M_FOCUS if focused else "")
                line.append(label, style=style)
            t.append_text(line)
            t.append("\n")
        t.append("\n")
        t.append(self._foot(), style=_M_DIM)
        self.query_one("#cm", Static).update(t)

    def _foot(self) -> str:
        if self.multi:
            return ("↑↓ move · Enter/Space — toggle ✓ (pick several) · type — your own option · "
                    "go to “✓ Done” + Enter to submit · Esc — skip")
        return "↑↓ move · type — your own answer · Enter — select · Esc — cancel"

    # ── навигация/выбор: BINDINGS-actions (надёжно без фокуса) ──────────────────
    def action_nav_up(self) -> None:
        self._focus = (self._focus - 1) % len(self._rows)
        self._redraw()

    def action_nav_down(self) -> None:
        self._focus = (self._focus + 1) % len(self._rows)
        self._redraw()

    def action_toggle(self) -> None:
        # Space: на «свой вариант» печатает пробел; на «Done» — отправляет; иначе ✓ (мультиселект).
        val = self._rows[self._focus][1]
        if val == _M_OTHER:
            self._other += " "
            self._redraw()
        elif val == _M_SUBMIT:
            self.dismiss(self._collect())
        elif self.multi:
            self._toggle(self._focus)

    def action_submit(self) -> None:
        # Enter. Мультиселект: на варианте — ПЕРЕКЛЮЧИТЬ ✓ (можно несколько!), на «Done»/«свой» —
        # отправить. Одиночный (Confirm): Enter — выбрать строку в фокусе и отправить.
        val = self._rows[self._focus][1]
        if not self.multi:
            self.dismiss(self._collect())
            return
        if val in (_M_SUBMIT, _M_OTHER):
            self.dismiss(self._collect())
        else:
            self._toggle(self._focus)

    def action_skip(self) -> None:
        self.dismiss(self._on_skip())

    # on_key — ТОЛЬКО свободный текст и цифры-хоткеи (нав/выбор/выход забрали BINDINGS).
    def on_key(self, event) -> None:
        on_other = self._rows[self._focus][1] == _M_OTHER
        ch = getattr(event, "character", None)
        if on_other:
            if event.key == "backspace":
                self._other = self._other[:-1]
                self._redraw()
                event.stop()
            elif ch is not None and ch.isprintable() and ch != " ":  # пробел — через action_toggle
                self._other += ch
                self._redraw()
                event.stop()
            return
        if ch and ch.isdigit():  # цифра — горячая отметка/фокус варианта
            idx = int(ch) - 1
            if 0 <= idx < len(self._rows) and self._rows[idx][1] != _M_OTHER:
                if self.multi:
                    self._toggle(idx)
                else:
                    self._focus = idx
                    self._redraw()
                event.stop()

    def _toggle(self, i: int) -> None:
        if self._rows[i][1] in (_M_OTHER, _M_SUBMIT):  # служебные строки не отмечаются
            return
        self._picked.discard(i) if i in self._picked else self._picked.add(i)
        self._redraw()

    def _collect(self) -> str:
        raise NotImplementedError

    def _on_skip(self) -> str:
        return ""


class ConfirmModal(_ChoiceModal):
    """HITL-подтверждение в стиле Claude Code: «да» / «да, всегда» / «нет» + свой ответ
    («да, но …»). Одиночный выбор: Enter — взять строку в фокусе (или свой текст)."""

    multi = False

    def __init__(self, description: str) -> None:
        rows = [("Yes", "yes"), ("Yes, always (stop asking)", "yes, always"),
                ("No", "no"), ("✎ your own answer", _M_OTHER)]
        super().__init__(description, rows, chip="⚠ Confirm")

    def _collect(self) -> str:
        free = self._other.strip()
        if free:
            return free
        label, value = self._rows[self._focus]
        return self._other.strip() if value == _M_OTHER else value

    def _on_skip(self) -> str:
        return "no"


class QuestionModal(_ChoiceModal):
    """Clarify Q/A в стиле Claude Code: чип «❓ Уточнение», нумерованные варианты с ❯/[✓],
    «✎ свой вариант» в списке, мультиселект. Ответ = выбранные + свой текст."""

    multi = True

    def __init__(self, question: str, options: list[str], why: str) -> None:
        rows = [(o, o) for o in options]
        rows.append(("your own option", _M_OTHER))
        rows.append(("Done — submit", _M_SUBMIT))  # Enter здесь — отправить выбранное
        super().__init__(question, rows, why=why, chip="❓ Clarify")

    def _collect(self) -> str:
        picked = [self._rows[i][0] for i in sorted(self._picked)
                  if self._rows[i][1] not in (_M_OTHER, _M_SUBMIT)]
        free = self._other.strip()
        return ", ".join(picked + ([free] if free else []))


class AttachModal(ModalScreen[str]):
    """Файловый ввод: путь к файлу для /attach (сессионное вложение). Возвращает путь."""

    CSS = """
    AttachModal { align: center middle; }
    #abox { width: 80%; max-width: 110; height: auto; border: round #a3be8c; padding: 1 2;
            background: $surface; }
    #atitle { color: #a3be8c; text-style: bold; }
    """
    BINDINGS = [("escape", "cancel", "Отмена")]

    def compose(self) -> ComposeResult:
        with Vertical(id="abox"):
            yield Label("📎 Attach a file to the session", id="atitle")
            yield Label("[dim]enter a path (pdf/image/audio/video/txt/csv…) · Enter · Esc — cancel[/]")
            yield Input(placeholder="/path/to/file", id="path")

    def on_mount(self) -> None:
        self.query_one("#path", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss((event.value or "").strip())


class VoiceModal(ModalScreen[str]):
    """Микрофонный ввод: пишет с мика (ffmpeg avfoundation) до Enter, расшифровывает → текст."""

    CSS = """
    VoiceModal { align: center middle; }
    #vbox { width: 60%; max-width: 80; height: auto; border: round #bf616a; padding: 1 2;
            background: $surface; align: center middle; }
    #vlbl { color: #bf616a; text-style: bold; }
    """
    BINDINGS = [("enter", "stop", "Стоп"), ("escape", "cancel", "Отмена")]

    def __init__(self) -> None:
        super().__init__()
        self._proc = None
        self._wav = None
        self._vtimer = None
        self._t0 = 0.0
        self._vphase = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="vbox"):
            yield Label("● Recording", id="vlbl")
            yield Label("[dim]speak · Enter — stop · Esc — cancel[/]")

    def on_mount(self) -> None:
        import shutil
        import subprocess
        import tempfile
        import time
        if not shutil.which("ffmpeg"):
            self.dismiss("__noffmpeg__")
            return
        self._wav = tempfile.mkstemp(suffix=".wav")[1]
        self._proc = subprocess.Popen(
            # -t 300: ffmpeg САМ остановится через 5 мин → страховка от утечки мика, если стоп
            # не вызвали (закрыли модалку/сбой). Иначе процесс держит микрофон бесконечно.
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "avfoundation", "-i", ":0",
             "-ac", "1", "-ar", "16000", "-t", "300", self._wav],
            stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._t0 = time.monotonic()
        self._vtimer = self.set_interval(0.14, self._vtick)  # живая индикация записи

    def _vtick(self) -> None:
        import random
        import time
        self._vphase += 1
        p = self._vphase
        el = int(time.monotonic() - self._t0)
        dot = "[bold red]●[/]" if p % 4 < 2 else "[red]○[/]"        # пульс
        bars = "".join(random.choice("▁▂▃▄▅▆▇") for _ in range(12))  # псевдо-waveform (уровень)
        self.query_one("#vlbl", Label).update(
            f"{dot} [bold]Запись[/] [dim]{el // 60}:{el % 60:02d}[/]   [#bf616a]{bars}[/]")

    def _stop_timer(self) -> None:
        if self._vtimer is not None:
            self._vtimer.stop()
            self._vtimer = None

    def _kill_proc(self) -> None:
        """SIGTERM → SIGKILL-фолбэк: гарантированно отпустить микрофон (ffmpeg порой игнорит TERM)."""
        if not self._proc:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                self._proc.kill()
                self._proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                pass

    def action_cancel(self) -> None:
        self._stop_timer()
        self._kill_proc()
        self.dismiss("")

    def action_stop(self) -> None:
        self._stop_timer()
        self.query_one("#vlbl", Label).update("⏳ Transcribing…")
        self._finish()

    @work(thread=True, exclusive=True)
    def _finish(self) -> None:
        from pathlib import Path as _P

        from src.media import transcribe_audio
        try:
            self._kill_proc()  # terminate → SIGKILL-фолбэк: мик отпущен гарантированно
            text = ""
            if self._wav and _P(self._wav).exists() and _P(self._wav).stat().st_size > 1000:
                text = (transcribe_audio(self._wav) or "").strip()
            if self._wav:
                _P(self._wav).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            text = ""
        self.app.call_from_thread(self.dismiss, text)  # call_from_thread — у App, не у Screen


class SeaTUI(App):
    """Чат-TUI агента: хедер-баннер · лента · рамка ввода · слэш-команды · HITL-модалки."""

    CSS = """
    Screen { layout: vertical; }
    #banner { height: auto; padding: 0 1; }
    #log { height: 1fr; min-height: 3; border: round #24405c; padding: 0 1;
           scrollbar-size: 0 0; }
    #status { height: 1; color: #7c899c; padding: 0 1; }
    #prompt { border: round #3a6ea5; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit"), ("ctrl+d", "quit", "Quit"),
                ("escape", "close_chat", "Close opened chat")]

    def action_close_chat(self) -> None:
        """Esc — закрыть открытый исторический чат и вернуться в свежую сессию (как /new)."""
        if not self._opened_chat:
            return
        self.thread_id = uuid.uuid4().hex
        self.history = []
        self._opened_chat = False
        self.query_one("#log", RichLog).clear()
        self._log("[#5b6472]✕ Closed chat — new session.[/]")
        self._focus_prompt()

    def __init__(self) -> None:
        super().__init__()
        self.graph = None
        self.history: list[dict] = []
        self.thread_id = "local"
        self._chat_list: list[dict] = []
        self._input_history: list[str] = []  # история ввода (↑/↓ как в шелле)
        self._hist_idx = 0
        self._cur_tracker = None  # трекер активного прогона (для токенов на лету)
        self._opened_chat = False  # открыт исторический чат (Esc — закрыть в свежий)
        self._ext_thread = uuid.uuid4().hex   # ветка истории для чата из браузерного расширения
        self._ext_history: list[dict] = []     # накопительная история панели расширения
        # Анимация «размышления»: фаза (перелив цвета + точка), текущее слово, лейбл ноды, таймер.
        self._think_phase = 0
        self._think_word = "Thinking"
        self._node_label = ""
        self._think_timer = None
        self._think_t0 = 0.0
        self._tok_in = 0
        self._tok_out = 0
        self._banner_phase = 0
        self._banner_timer = None
        self._welcome = True  # приветственный баннер показан (удаляется при первой отправке)
        self._agent_loop = None  # постоянный event-loop для графа (отдельный поток; UI не виснет)

    def compose(self) -> ComposeResult:
        yield Static(_banner(), id="banner")
        yield RichLog(id="log", wrap=True, markup=True, highlight=True)
        yield Static("", id="status")
        inp = Input(placeholder="task · /help — commands · “/” — suggestions · exit — quit",
                    id="prompt", suggester=SuggestFromList(_COMMANDS, case_sensitive=False))
        inp.border_title = "❯"
        yield inp

    async def on_mount(self) -> None:
        import asyncio
        import threading
        # Мышь ОСТАЁТСЯ за приложением → колесо скроллит ленту. Выделение мышью — родным механизмом
        # Textual (ALLOW_SELECT=True): тянешь по тексту — выделяется и копируется. Для нативного
        # выделения терминала (если нужно) — ⌥-drag (Option). (Раньше я отключал мышь → сломал скролл.)
        self.query_one("#prompt", Input).focus()
        self._status("🌊 warming up the agent (models/skills)… [dim](this welcome disappears after your first message)[/]")
        # ПРИВЕТСТВЕННЫЙ ЭКРАН = баннер (лого SEA + Ика + инструкции). Лента чата ЧИСТАЯ (без Ики!).
        # Баннер ВРЕМЕННЫЙ: удаляется при первой отправке (_dismiss_welcome) → чат на весь экран.
        self._banner_timer = self.set_interval(0.22, self._banner_tick)  # перелив лого, пока виден баннер
        # Постоянный event-loop графа в фоновом потоке: LLM-httpx-клиент привязан к нему (без
        # «loop closed» между запросами), а UI-loop остаётся свободным (приложение не виснет).
        self._agent_loop = asyncio.new_event_loop()
        threading.Thread(target=self._agent_loop.run_forever, daemon=True, name="sea-agent-loop").start()
        self._build_graph()

    def _banner_tick(self) -> None:
        self._banner_phase += 1
        try:
            self.query_one("#banner", Static).update(_banner(self._banner_phase))
        except Exception:  # noqa: BLE001
            pass

    def _dismiss_welcome(self) -> None:
        """Убрать приветственный баннер (лого+Ика+инструкции) при первой отправке — чат на весь
        экран. /help вернёт инструкции в ленту. Идемпотентно."""
        if not self._welcome:
            return
        self._welcome = False
        try:
            if self._banner_timer is not None:
                self._banner_timer.stop()
                self._banner_timer = None
            self.query_one("#banner", Static).remove()
        except Exception:  # noqa: BLE001
            pass

    # ── инфраструктура ───────────────────────────────────────────────
    def _log(self, renderable) -> None:
        # expand=True — растянуть рендер на ВСЮ ширину ленты (RichLog по умолчанию expand=False →
        # подгоняет панель под ширину текста, а не окна; из-за этого панели были не во всю ширину).
        self.query_one("#log", RichLog).write(renderable, expand=True)

    def _status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    # ── анимация «размышления» (перелив синий↔металлик + точечная фигурка + крутящиеся слова) ──
    def _start_thinking(self) -> None:
        import random
        import time

        from src import cli_art
        self._think_phase = 0
        self._node_label = ""
        self._think_t0 = time.monotonic()  # настенные часы прогона (runbudget.elapsed() — thread-local → 0 на UI-треде)
        self._think_word = random.choice(cli_art.THINKING_WORDS)
        if self._think_timer is not None:
            self._think_timer.stop()
        self._think_timer = self.set_interval(0.12, self._tick)

    def _stop_thinking(self) -> None:
        if self._think_timer is not None:
            self._think_timer.stop()
            self._think_timer = None

    def _tick(self) -> None:
        import random

        from src import cli_art
        self._think_phase += 1
        p = self._think_phase
        if p % 18 == 1:  # менять слово ~раз в 2с
            self._think_word = random.choice(cli_art.THINKING_WORDS)
        color = cli_art.shimmer_color(p)          # перелив синий↔металлик-серый (как лого SEA)
        dot = cli_art.think_dot(p)                # точечная фигурка (брайль), морфится
        # Лейбл ноды (прогресс шага) — ВИДИМЫЙ (не dim), чтобы было ясно, что агент работает,
        # а не завис: «🛠 step 2/3 done», «🧠 reasoning» и т.п. + сколько секунд идёт прогон.
        import time
        secs = int(time.monotonic() - self._think_t0)  # настенные часы (НЕ runbudget — он thread-local → 0)
        nl = f"  [#88c0d0]{self._node_label}[/]" if self._node_label else ""
        # Токены НА ЛЕТУ: сессия + текущий прогон (трекер растёт после каждого LLM-вызова).
        ci = self._cur_tracker.input if self._cur_tracker else 0
        co = self._cur_tracker.output if self._cur_tracker else 0
        li, lo = self._tok_in + ci, self._tok_out + co
        # «вызов в полёте»: started>calls → модель СЕЙЧАС отвечает (токены придут пачкой на завершении,
        # поэтому 0 tok при идущем вызове — норма, не простой). Точка-пульс показывает живость.
        inflight = (self._cur_tracker.started - self._cur_tracker.calls) if self._cur_tracker else 0
        flight = "  [#88c0d0]⟳ calling model…[/]" if inflight > 0 else ""
        self._status(f"[{color}]{dot} {self._think_word}…[/]{nl}  [dim]{secs}s · 🧮 ↓{li} ↑{lo} "
                     f"({li + lo} tok)[/]{flight}")

    def _on_node_label(self, lbl: str) -> None:
        # Что агент делает сейчас: в статус (анимация) И трейлом в ленту (видно «размышления»/шаги).
        self._node_label = lbl
        self._log(f"[dim cyan]· {lbl}[/]")

    def on_key(self, event) -> None:
        try:
            inp = self.query_one("#prompt", Input)
        except Exception:  # noqa: BLE001
            return
        if not inp.has_focus:
            return
        # Tab — ПРИНЯТЬ подсказку команды (ghost-text от suggester) = автодополнение.
        if event.key == "tab":
            sug = getattr(inp, "_suggestion", "") or ""
            if sug and sug != inp.value:
                inp.value = sug
                inp.cursor_position = len(sug)
                event.prevent_default()
                event.stop()
        # ↑/↓ — история ввода (как в шелле): прокрутка ранее отправленных сообщений/команд.
        elif event.key == "up" and self._input_history:
            if self._hist_idx > 0:
                self._hist_idx -= 1
                inp.value = self._input_history[self._hist_idx]
                inp.cursor_position = len(inp.value)
            event.prevent_default()
            event.stop()
        elif event.key == "down" and self._input_history:
            if self._hist_idx < len(self._input_history) - 1:
                self._hist_idx += 1
                inp.value = self._input_history[self._hist_idx]
            else:  # ниже последней — пустая строка (новый ввод)
                self._hist_idx = len(self._input_history)
                inp.value = ""
            inp.cursor_position = len(inp.value)
            event.prevent_default()
            event.stop()

    @work(thread=True, exclusive=True)
    def _build_graph(self) -> None:
        """Прогрев агента в ОТДЕЛЬНОМ потоке (UI не виснет). HITL-каналы → модалки TUI."""
        from langgraph.checkpoint.memory import MemorySaver

        from src import clarify, hitl
        from src.agent import build_graph
        from src.cli_config import get_cli

        hitl.load_grants(get_cli("allow") or [])
        hitl.set_work_mode(get_cli("work_mode") or "auto-accept")
        # HITL/clarify через модалки (вызовутся из графового воркера, тот в loop приложения).
        hitl.set_confirmer(self._confirm)
        clarify.set_clarifier(self._clarify)
        graph = build_graph(MemorySaver())
        self.call_from_thread(self._graph_ready, graph)

    def _graph_ready(self, graph) -> None:
        from src import hitl
        self.graph = graph
        self._status(f"ready · mode: {hitl.work_mode()}")
        # Первый запуск без ключа (новый проект, установлен пакетом) — мягкая подсказка, не падаем.
        try:
            from src.llm import api_key
            if not api_key():
                self._log("[#d08770]⚠ No API key set.[/] Run [bold]/key <API_KEY>[/] (saved globally, "
                          "works in every project) · [bold]/provider openrouter|ollama[/] to switch.")
        except Exception:  # noqa: BLE001
            pass
        self._start_bridge()  # поднять мост браузерного расширения (как старый REPL)

    def _start_bridge(self) -> None:
        """Мост к браузерному расширению (агент в ТВОЁМ браузере + чат из side-panel). Раньше его
        поднимал REPL; в TUI его не было → расширение «перестало подниматься». Поднимаем МОЛЧА —
        без спама токеном на старте (его достаёт команда /token, когда реально нужно подключать
        расширение). Сервер моста крутит свой loop в своём потоке."""
        try:
            from src import browser_bridge
            browser_bridge.ensure_server()
            browser_bridge.set_chat_handler(self._ext_chat)
        except Exception as e:  # noqa: BLE001
            self._log(f"[#bf616a]browser bridge failed:[/] {e}")

    async def _ext_chat(self, text: str) -> str:
        """Чат из side-panel расширения → тот же граф (своя ветка истории). Зовётся на loop'е
        моста; граф гоняем на agent-loop (там живут LLM-клиенты) через run_coroutine_threadsafe."""
        import asyncio

        from src import run_context, runbudget
        from src.usage import TokenTracker, cost_of
        tracker = TokenTracker()
        rid = uuid.uuid4().hex

        async def _run():
            with run_context.request_scope(rid, "local"):
                runbudget.reset()
                res = await self.graph.ainvoke(
                    {"query": text, "user_id": "local", "session_id": self._ext_thread,
                     "force_mode": "", "chat_history": self._ext_history + [{"role": "user", "content": text}]},
                    config={"configurable": {"thread_id": self._ext_thread}, "recursion_limit": 50,
                            "callbacks": [tracker]})
                return res.get("final_answer", "") or "(empty answer)"
        try:
            fut = asyncio.run_coroutine_threadsafe(_run(), self._agent_loop)
            ans = await asyncio.wrap_future(fut)
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"
        self._ext_history += [{"role": "user", "content": text}, {"role": "assistant", "content": ans}]
        del self._ext_history[:-20]
        di, do = tracker.input, tracker.output
        self.call_from_thread(
            self._log, f"[#5b6472]🧩 extension chat: “{text[:40]}” · {di + do} tok (~${cost_of(di, do):.4f})[/]")
        return f"{ans}\n\n🧮 {di + do} tok (~${cost_of(di, do):.4f})"

    # ── HITL/clarify каналы → модалка на app-loop, БЛОКИРУЯ графовый ПОТОК (бридж) ─────
    # Граф крутится в отдельном потоке (UI не виснет), а push_screen — операция app-loop.
    # Поэтому показываем модалку через call_from_thread и ждём ответ threading.Event'ом.
    def _confirm(self, description: str) -> str:
        import threading
        box, ev = {}, threading.Event()

        def _show():
            self.push_screen(ConfirmModal(description),
                             lambda r: (box.__setitem__("v", r), ev.set()))
        self.app.call_from_thread(_show)
        ev.wait()
        return box.get("v", "нет")

    def _clarify(self, items: list[dict]) -> list[str]:
        import threading
        out: list[str] = []
        for it in items:
            box, ev = {}, threading.Event()

            def _show(_it=it):
                self.push_screen(
                    QuestionModal(_it.get("question", ""), list(_it.get("options", []) or []),
                                  _it.get("why", "")),
                    lambda r: (box.__setitem__("v", r), ev.set()))
            self.app.call_from_thread(_show)
            ev.wait()
            out.append(box.get("v", ""))
        return out

    # ── ввод ──────────────────────────────────────────────────────────
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        text = (event.value or "").strip()
        if not text:
            return
        if text.lower() in ("exit", "quit", "q"):
            self.exit()
            return
        self.query_one("#prompt", Input).value = ""
        # История ввода (↑/↓): добавляем отправленное (без подряд-дублей), курсор — в конец.
        # БЕЗОПАСНОСТЬ: команды с секретом (/key <API_KEY>) в историю НЕ кладём — иначе ↑ показал бы ключ.
        _secret = text.lower().startswith(("/key", "/login"))
        if not _secret and (not self._input_history or self._input_history[-1] != text):
            self._input_history.append(text)
        self._hist_idx = len(self._input_history)
        self._dismiss_welcome()  # первая отправка → убрать приветственный баннер (лого+Ика+инструкции)
        if text.startswith("/"):
            # В ВОРКЕРЕ: некоторые команды (/voice, /attach) зовут push_screen_wait, а он требует
            # активного воркера (NoActiveWorker иначе — баг ревью). Обработчик события — не воркер.
            self._run_command(text)
            return
        if self.graph is None:
            self._status("still warming up — one sec…")
            return
        self._log(f"\n[bold #38c6ff]❯ you[/]\n{text}")  # БЕЗ рамки (рамка только для SEA.md)
        self._ask(text)

    # ── слэш-команды ──────────────────────────────────────────────────
    @work(exclusive=False)
    async def _run_command(self, raw: str) -> None:
        """Команды — в ВОРКЕРЕ (push_screen_wait у /voice//attach требует активного воркера).
        Любая ошибка команды — В ЛЕНТУ (а не молча проглатывается), затем фокус назад в поле ввода."""
        try:
            await self._command(raw)
        except Exception as e:  # noqa: BLE001
            self._log(f"[red]✗ {raw}:[/] {type(e).__name__}: {e}")
        finally:
            self._focus_prompt()

    async def _command(self, raw: str) -> None:
        parts = raw.split()
        cmd, args = parts[0].lower(), parts[1:]
        from src import hitl
        if cmd in ("/help", "/?"):
            self._log(
                "\n[bold #38c6ff]commands[/]\n"
                "[bold]/help[/] · /clear · /new · /init · exit\n"
                "[bold]/auto[/] [accept|off|plan|full] — work mode\n"
                "[bold]/model[/] [api|ollama <name>] — provider/model\n"
                "[bold]/chats[/] [N] — open · /chats del N — delete · /chats clear — all · "
                "[bold]/fav[/] — ★ current\n"
                "[bold]/facts[/] — what the agent remembers · [bold]/goal[/] — current goal\n"
                "[bold]/usage[/] — token spend · [bold]/traces[/] — node stats\n"
                "[bold]/diagnose[/] — self-diagnostics · [bold]/improve[/] — self-learning\n"
                "[bold]/attach[/] [path] — file into session · [bold]/voice[/] — voice input 🎙\n"
                "[bold]/kb[/] add|ls|find — knowledge base · [bold]/compact[/] · [bold]/sync[/]\n"
                "[bold]/token[/] — show browser-extension bridge token\n"
                "[dim]↑/↓ — input history · Tab — accept suggestion · Esc — close opened chat · "
                "drag the mouse — select & copy[/]")
        elif cmd == "/clear":
            self.query_one("#log", RichLog).clear()
        elif cmd == "/new":
            self.thread_id = uuid.uuid4().hex
            self.history = []
            self._opened_chat = False
            self.query_one("#log", RichLog).clear()
            self._log("[#5b6472]New thread.[/]")
        elif cmd == "/init":
            from src.sea_workspace import init
            created = init()
            self._log("[#5b6472]Workspace:[/] " + (", ".join(created) if created else "already initialized"))
            # /init like Claude Code: not just scaffold — actually analyze the repo (LLM reads
            # README/manifests/docstrings) and write a meaningful SEA.md. Runs in background.
            self._log("[dim]🔎 Analyzing repository → SEA.md (overview, architecture, modules)…[/]")
            self._do_init_overview()
        elif cmd == "/auto":
            mode = {"off": "manual", "accept": "auto-accept", "full": "auto",
                    "plan": "plan"}.get(args[0] if args else "accept", "auto-accept")
            from src.cli_config import set_cli
            hitl.set_work_mode(mode)
            set_cli("work_mode", mode)
            self._status(f"mode: {hitl.work_mode()}")
            self._log(f"[#5b6472]Work mode: {hitl.work_mode()}[/]")
        elif cmd == "/model":
            from src.cli_config import set_cli
            from src.llm import active_summary, set_provider
            from src.agent import rebuild_llms
            if args and args[0] == "ollama":
                set_provider("ollama", args[1] if len(args) > 1 else None)
                set_cli("provider", "ollama")
            else:
                set_provider("openrouter", None)
                set_cli("provider", "openrouter")
            rebuild_llms()
            self._log(f"[#5b6472]Model: {active_summary()}[/]")
        elif cmd == "/key":
            # Ключ провайдера → ГЛОБАЛЬНЫЙ ~/.config/sea/config.local.yml (работает во всех проектах).
            from src.cli_config import set_cli
            from src.agent import rebuild_llms
            if not args:
                self._log("[#bf616a]Usage:[/] /key <API_KEY>")
            else:
                set_cli("api_key", args[0].strip())
                rebuild_llms()
                self._log("[#5b6472]✓ API key saved (user config) — works in every project.[/]")
        elif cmd == "/provider":
            from src.cli_config import set_cli
            from src.llm import active_summary, set_provider
            from src.agent import rebuild_llms
            name = args[0] if args and args[0] in ("openrouter", "ollama") else None
            if not name:
                self._log("[#bf616a]Usage:[/] /provider openrouter|ollama [base_url]")
            else:
                set_provider(name, None)
                set_cli("provider", name)
                if len(args) > 1:
                    set_cli("base_url", args[1].strip())
                rebuild_llms()
                self._log(f"[#5b6472]✓ Provider: {name}[/] · {active_summary()}")
        elif cmd == "/config":
            from src import config_paths
            from src.cli_config import get_cli
            from src.llm import active_summary, api_key_source
            self._log("\n[bold #38c6ff]config[/]\n"
                      f"provider : {get_cli('provider') or 'openrouter (default)'}\n"
                      f"models   : {active_summary()}\n"
                      f"api key  : {api_key_source()}\n"
                      f"base cfg : {config_paths.base_config_path()}\n"
                      f"user cfg : {config_paths.global_local_path()}")
        elif cmd == "/chats":
            from src import chat_store
            sub = args[0].lower() if args else ""
            if args and args[0].isdigit():  # /chats N — open
                idx = int(args[0]) - 1
                if 0 <= idx < len(self._chat_list):
                    t = self._chat_list[idx]
                    self.thread_id = t["thread_id"]
                    self.history = chat_store.get_messages(self.thread_id, last=20)
                    self.query_one("#log", RichLog).clear()
                    self._dismiss_welcome()
                    self._opened_chat = True  # Esc закроет этот чат → свежая сессия
                    self._log(f"[#5b6472]↩ Opened chat: {t.get('title','?')}  [dim](Esc — close)[/][/]")
                    # ОТРИСОВАТЬ загруженную переписку (раньше история грузилась, но не показывалась).
                    for m in self.history:
                        role, content = m.get("role", ""), str(m.get("content", ""))
                        if role == "user":
                            self._log(f"\n[bold #38c6ff]❯ you[/]\n{content}")
                        elif role == "assistant":
                            self._log(f"\n[bold #38c6ff]🌊 agent[/]\n{content}")
                    if not self.history:
                        self._log("[dim](this chat has no messages)[/]")
                else:
                    self._log("[#bf616a]No such number.[/]")
            elif sub in ("del", "rm", "delete") and len(args) > 1 and args[1].isdigit():
                idx = int(args[1]) - 1
                if 0 <= idx < len(self._chat_list):
                    t = self._chat_list[idx]
                    chat_store.delete_thread(t["thread_id"])
                    self._log(f"[#5b6472]🗑 Deleted chat:[/] {t.get('title','?')}")
                    self._chat_list = chat_store.list_threads("local", limit=20)  # renumber
                else:
                    self._log("[#bf616a]No such number. Run /chats first for the list.[/]")
            elif sub in ("clear", "purge"):
                n = 0
                for t in chat_store.list_threads("local", limit=1000):
                    chat_store.delete_thread(t["thread_id"])
                    n += 1
                self._chat_list = []
                self._log(f"[#5b6472]🗑 Deleted chats: {n}[/]")
            else:
                self._chat_list = chat_store.list_threads("local", limit=20)
                if not self._chat_list:
                    self._log("[dim]No saved chats yet.[/]")
                else:
                    lines = "\n".join(f"  {i+1}. {t.get('title','?')} "
                                      f"[dim]({t.get('msg_count',0)//2} turns)[/]"
                                      for i, t in enumerate(self._chat_list))
                    self._log("\n[bold #38c6ff]chats[/]\n" + lines
                              + "\n[dim]/chats N — open · /chats del N — delete · /chats clear — delete all[/]")
        elif cmd == "/diagnose":
            from src.agent import memory_store
            from src.tracing import diagnose
            rep = diagnose(memory_store, self.thread_id)  # память скоупится по чату (thread)
            self._log("\n[bold #38c6ff]diagnose[/]\n" + str(rep))
        elif cmd == "/facts":
            from src.agent import memory_store
            facts = memory_store.get_facts(self.thread_id)  # факты ЭТОГО чата (изоляция по треду)
            if not facts:
                self._log("[dim]Nothing remembered in this chat yet.[/]")
            else:
                self._log("\n[bold #38c6ff]what I remember (this chat)[/]\n"
                          + "\n".join(f"• [bold]{f['key']}[/]: {f['value']}" for f in facts[:30]))
        elif cmd == "/goal":
            from src.agent import memory_store
            if args and args[0].lower() in ("clear", "done", "reset", "forget"):
                # Закрыть активную цель ЭТОГО чата (память скоупится по треду — изоляция между чатами).
                g = memory_store.get_active_goal(self.thread_id)
                if g:
                    memory_store.close_active_goal(self.thread_id)
                    self._log(f"[#5b6472]🎯 Goal cleared:[/] {g['aim']}")
                else:
                    self._log("[dim]No active goal to clear.[/]")
            else:
                g = memory_store.get_active_goal(self.thread_id)
                if not g:
                    self._log("[dim]No active goal.[/]")
                else:
                    crit = memory_store.goal_criteria(g)
                    txt = f"🎯 {g['aim']}" + ("\n" + "\n".join(f"  ☐ {c}" for c in crit) if crit else "")
                    self._log("\n[bold #38c6ff]goal[/] [dim](/goal clear — forget it)[/]\n" + txt)
        elif cmd == "/usage":
            from src.usage import cost_of
            cost = cost_of(self._tok_in, self._tok_out)
            self._log(f"\n[bold #38c6ff]session spend[/]\nin: {self._tok_in} · out: {self._tok_out} · "
                      f"total: {self._tok_in + self._tok_out} tok (~${cost:.4f})")
        elif cmd == "/traces":
            from src.tracing import trace_store
            rows = trace_store.node_stats(24.0)
            if not rows:
                self._log("[dim]No traces yet.[/]")
            else:
                self._log("\n[bold #38c6ff]node traces (24h)[/]\n"
                          + "\n".join(f"{dict(r).get('node','?')}: {dict(r)}" for r in rows[:20]))
        elif cmd == "/fav":
            from src import chat_store
            if not chat_store.get_thread(self.thread_id):
                self._log("[dim]Chat not saved yet (no exchange).[/]")
            else:
                new = chat_store.toggle_favorite(self.thread_id)
                self._log(f"[#5b6472]★ favorite: {'on' if new else 'off'}[/]")
        elif cmd == "/improve":
            self._log("[dim]Running self-learning (backward) in background…[/]")
            self._run_improve()
        elif cmd == "/attach":
            path = " ".join(args) if args else await self.push_screen_wait(AttachModal())
            if path:
                self._attach_file(path)
        elif cmd == "/voice":
            text = await self.push_screen_wait(VoiceModal())
            if text == "__noffmpeg__":
                self._log("[#bf616a]ffmpeg required: brew install ffmpeg[/]")
            elif text:
                self.query_one("#prompt", Input).value = text
                self._log(f"[dim]🎙 recognized: “{text[:80]}” — press Enter to send.[/]")
            else:
                self._log("[dim]Empty (silence / no microphone access).[/]")
        elif cmd == "/kb":
            sub = args[0].lower() if args else "ls"
            if sub == "add" and len(args) > 1:
                self._kb_add(" ".join(args[1:]))
            elif sub == "find" and len(args) > 1:
                self._kb_find(" ".join(args[1:]))
            else:
                self._kb_ls()
        elif cmd in ("/token", "/bridge"):
            from src import browser_bridge as _bb
            self._log(f"[#5b6472]🧩 Browser bridge 127.0.0.1:{_bb.PORT} · serving: "
                      f"{getattr(_bb,'_serving',False)} · token:[/] [#38c6ff]{_bb.token()}[/]")
        elif cmd in ("/compact", "/compress"):
            self._do_compact()
        elif cmd == "/sync":
            self._do_sync()
        else:
            self._log(f"[#bf616a]Unknown command {cmd}[/] — /help")

    # ── прогон графа — в ОТДЕЛЬНОМ ПОТОКЕ (UI не виснет; граф мешает sync+async) ────────
    @work(thread=True, exclusive=False)
    def _ask(self, query: str) -> None:
        import asyncio

        from src import chat_store, hitl, run_context, runbudget
        from src.progress import stream_with_progress
        from src.usage import TokenTracker

        call = self.app.call_from_thread  # обновления UI — только через app-loop
        tracker = TokenTracker()
        self._cur_tracker = tracker  # текущий трекер → _tick читает токены НА ЛЕТУ (растут по ходу прогона)
        call(self._start_thinking)
        rid = uuid.uuid4().hex

        async def _run():
            # request_scope/reset — ВНУТРИ корутины (на agent-loop), чтобы run_id видели ноды графа.
            with run_context.request_scope(rid, "local"):
                runbudget.reset()
                return await stream_with_progress(
                    self.graph,
                    {"query": query, "user_id": "local", "session_id": self.thread_id,
                     "force_mode": "", "chat_history": self.history + [{"role": "user", "content": query}]},
                    config={"configurable": {"thread_id": self.thread_id}, "recursion_limit": 50,
                            "callbacks": [tracker]},
                    on_label=lambda lbl: call(self._on_node_label, lbl))
        try:
            fut = asyncio.run_coroutine_threadsafe(_run(), self._agent_loop)
            result = fut.result()  # блокирует ПОТОК воркера (не UI), граф крутится на agent-loop
            ans = result.get("final_answer", "") or "(empty answer)"
        except Exception as e:  # noqa: BLE001
            ans = f"[#bf616a]error:[/] {type(e).__name__}: {e}"
        finally:
            call(self._stop_thinking)
            self._cur_tracker = None  # прогон завершён — дальше статус показывает сессионные итоги
        self.history += [{"role": "user", "content": query}, {"role": "assistant", "content": ans}]
        self.history[:] = self.history[-20:]
        try:
            chat_store.record_turn(self.thread_id, "local", query, ans)
        except Exception:  # noqa: BLE001
            pass
        call(self._log, f"\n[bold #38c6ff]🌊 agent[/]\n{ans}")  # БЕЗ рамки (рамка только для SEA.md)
        di, do = tracker.input, tracker.output
        self._tok_in += di
        self._tok_out += do
        from src.usage import cost_of
        cost = cost_of(di, do)
        # Видимая в ЛЕНТЕ строка токенов за этот ход (вход/выход) + накопительно за сессию.
        call(self._log, f"[dim]🧮 in {di} · out {do} · turn {di + do} · session {self._tok_in + self._tok_out} tok "
                        f"(~${cost:.4f})[/]")
        call(self._status, f"ready · mode: {hitl.work_mode()} · 🧮 ↓{self._tok_in} ↑{self._tok_out} "
                           f"({self._tok_in + self._tok_out} tok)")
        call(self._focus_prompt)  # вернуть фокус в поле ввода (после модалок фокус мог уйти → «команды не работали»)

    def _focus_prompt(self) -> None:
        try:
            self.query_one("#prompt", Input).focus()
        except Exception:  # noqa: BLE001
            pass

    @work(thread=True, exclusive=True)
    def _run_improve(self) -> None:
        from src.agent import memory_store
        from src.improve import graph_backward
        try:
            res = graph_backward(memory_store, 3)
            self.call_from_thread(self._log, f"[#5b6472]Self-learning:[/] {res.get('status', '?')} {res.get('reason', '')}")
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._log, f"[#bf616a]improve failed:[/] {type(e).__name__}: {e}")

    # ── файлы / KB ────────────────────────────────────────────────────
    @work(thread=True, exclusive=False)
    def _attach_file(self, path: str) -> None:
        from pathlib import Path as _P

        from src.knowledge_base import add_session_file
        try:
            stored = add_session_file(self.thread_id, str(_P(path).expanduser()))  # ПУТЬ к копии (не сообщение)
            if stored.startswith("["):  # «[файл не найден: …]»
                self.call_from_thread(self._log, f"[#bf616a]attach failed:[/] {stored}")
            else:  # файл проиндексирован в session-KB (recall-AutoRAG его подхватит)
                self.call_from_thread(self._log, f"[#5b6472]📎 attached:[/] {_P(stored).name}")
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._log, f"[#bf616a]attach failed:[/] {type(e).__name__}: {e}")

    @work(thread=True, exclusive=False)
    def _kb_add(self, path: str) -> None:
        from pathlib import Path as _P

        from src.knowledge_base import add_document
        try:
            msg = add_document("local", str(_P(path).expanduser()))
            self.call_from_thread(self._log, f"[#5b6472]📚 KB:[/] {msg}")
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._log, f"[#bf616a]kb add failed:[/] {type(e).__name__}: {e}")

    @work(thread=True, exclusive=False)
    def _kb_find(self, query: str) -> None:
        from src.knowledge_base import search_kb
        try:
            res = (search_kb("local", query, 5) or "(nothing found)")[:1500]
            self.call_from_thread(self._log, "\n[bold #38c6ff]KB[/]\n" + res)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._log, f"[#bf616a]kb find failed:[/] {type(e).__name__}: {e}")

    def _kb_ls(self) -> None:
        from pathlib import Path as _P
        try:
            from src.knowledge_base import _user_root
            root = _user_root("local")
            files = [p.name for p in _P(root).rglob("*") if p.is_file()][:40] if _P(str(root)).exists() else []
            self._log("[bold]KB:[/] " + (", ".join(files) if files else "empty")
                      + "\n[dim]/kb add <file> · /kb find <query>[/]")
        except Exception as e:  # noqa: BLE001
            self._log(f"[#bf616a]kb ls failed:[/] {type(e).__name__}: {e}")

    # ── compact / sync ────────────────────────────────────────────────
    @work(exclusive=False)
    async def _do_compact(self) -> None:
        from src import chat_store, compact as _cp, hitl
        from src.agent import llm
        msgs = chat_store.get_messages(self.thread_id)
        if len(msgs) < 2 and len(self.history) < 2:
            self._log("[dim]Context is still too small to compact.[/]")
            return
        convo = ("\n".join(f"{m['role']}: {m['content'][:800]}" for m in msgs)
                 or "\n".join(f"{m['role']}: {str(m.get('content', ''))[:800]}" for m in self.history))
        self._status("🗜 Compacting context…")
        try:
            resp = await llm.ainvoke(_cp.build_compact_messages(convo, _cp.read_compact(), _cp.gather_meta()))
            summary = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            n = _cp.append_compact(summary)
            chat_store.set_summary(self.thread_id, summary)
            self.history = [{"role": "assistant", "content": f"[COMPACT.md · compaction #{n}]"}] \
                + self.history[-_cp._KEEP_LAST_SCOPE:]
            self._log(f"\n[bold #38c6ff]compact[/]\n🗜 Context compacted → COMPACT.md (#{n})\n{summary[:400]}")
        except Exception as e:  # noqa: BLE001
            self._log(f"[#bf616a]Compaction failed:[/] {type(e).__name__}: {e}")
        self._status(f"mode: {hitl.work_mode()}")

    @work(exclusive=False)
    async def _do_init_overview(self) -> None:
        """LLM-обзор репозитория → SEA.md (как `/init` в Claude Code). Дайджест (README/манифесты/
        докстринги) детерминирован; модель пишет содержательный обзор. Ошибка/пусто → не трогаем скаффолд."""
        from pathlib import Path as _P

        from src.agent import llm
        from src.sea_workspace import build_overview_messages, gather_repo_digest
        self._status("🔎 Analyzing repository…")
        try:
            digest = gather_repo_digest()
            sea = _P("SEA.md")
            cur = sea.read_text(encoding="utf-8") if sea.exists() else ""
            resp = await llm.ainvoke(build_overview_messages(digest, cur))
            text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            if len(text) < 80:
                self._log("[dim]/init: model returned empty — kept the starter SEA.md map.[/]")
            else:
                sea.write_text(text + "\n", encoding="utf-8")
                # ЕДИНСТВЕННАЯ рамка в интерфейсе — описание SEA.md (тёмно-синяя, на всю ширину окна).
                self._log(Panel(text[:600] + ("\n[dim]… (full text in SEA.md)[/]" if len(text) > 600 else ""),
                                title="📄 SEA.md — project overview", title_align="left",
                                border_style="#24405c", expand=True))
        except Exception as e:  # noqa: BLE001
            self._log(f"[#bf616a]/init overview failed:[/] {type(e).__name__}: {e} [dim](starter map kept)[/]")
        self._status("ready")

    @work(exclusive=False)
    async def _do_sync(self) -> None:
        from pathlib import Path as _P

        from src import compact as _cp
        from src.agent import llm
        from src.sea_workspace import scan_repo
        ct = _cp.read_compact()
        if not ct.strip():
            self._log("[dim]COMPACT.md is empty — run /compact first, then /sync.[/]")
            return
        sea = _P("SEA.md")
        cur = sea.read_text(encoding="utf-8") if sea.exists() else ""
        self._status("🔄 Rebuilding SEA.md…")
        try:
            resp = await llm.ainvoke(_cp.build_sea_rebuild_messages(ct, cur, scan_repo()))
            new_sea = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        except Exception as e:  # noqa: BLE001
            self._log(f"[#bf616a]/sync failed:[/] {type(e).__name__}: {e}")
            return
        if len(new_sea) < 40:
            self._log("[dim]/sync: model returned empty — SEA.md untouched.[/]")
            return
        sea.write_text(new_sea + "\n", encoding="utf-8")
        self._log("[#5b6472]🔄 SEA.md rebuilt from COMPACT.md[/]")
        self._status("ready")


def run_tui() -> None:
    """Точка входа TUI (`sea` по умолчанию)."""
    SeaTUI().run()
