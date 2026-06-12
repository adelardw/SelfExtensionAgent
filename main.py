import warnings
# Тихий старт: глушим шумные сторонние warnings ДО тяжёлых импортов.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
warnings.filterwarnings("ignore", message="urllib3")  # RequestsDependencyWarning о версиях

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from omegaconf import OmegaConf
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from src.agent import build_graph, config, memory_store, rebuild_llms
from src.clarify import set_clarifier
from src.hitl import set_confirmer
from src.llm import active_summary, set_provider
from src.media import AUDIO_EXTS, DOC_EXTS, IMAGE_EXTS, TEXT_EXTS, attachment_context, transcribe_audio
from src.knowledge_base import (
    add_document_async, add_session_file, create_folder, list_kb, clear_session, search_kb_async,
)
from src.progress import stream_with_progress
from src.tracing import diagnose, trace_store
from src.usage import TokenTracker, add_alltime, cost_of, load_alltime

console = Console()

# Ссылка на активный спиннер прогресса. Запрос ввода (HITL-подтверждение, уточнения)
# ДОЛЖЕН останавливать спиннер: Rich Live владеет терминалом и затирает строку ввода,
# из-за чего пользователь физически не может ответить (баг: подтверждение не проходило).
_live_status = None


async def _paused_input(prompt: str = "") -> str:
    """Ввод с паузой спиннера прогресса (иначе Rich Live перебивает запрос → ввод не проходит)."""
    st = _live_status
    if st is not None:
        try:
            st.stop()
        except Exception:  # noqa: BLE001
            pass
    try:
        return await asyncio.to_thread(input, prompt)
    finally:
        if st is not None:
            try:
                st.start()
            except Exception:  # noqa: BLE001
                pass

# Цвета режимов мышления (Self-Reflexion Choice).
MODE_STYLE = {
    "fast": ("⚡ FAST", "bold green"),
    "reason": ("🧠 REASON", "bold cyan"),
    "deliberate": ("🛠 DELIBERATE", "bold yellow"),
    "heavy": ("🏗 HEAVY", "bold red"),
    "clarify": ("❓ CLARIFY", "bold magenta"),
}


@asynccontextmanager
async def make_checkpointer():
    backend = config.get("checkpointer", {}).get("backend", "memory")
    if backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = config.checkpointer.get("sqlite_path", "data/checkpoints.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            yield saver
    else:
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()


def banner() -> None:
    legend = "  ".join(f"[{st}]{lbl}[/]" for lbl, st in MODE_STYLE.values())
    body = Text.from_markup(
        f"Активно: [cyan]{active_summary()}[/]\n"
        f"Режимы: {legend}\n"
        f"Команды: [dim]/model /voice /help /new /facts /goal /diagnose /traces /improve /usage  ·  exit[/]\n"
        f"Файлы: [dim]/attach <файл> — в сессию (tmp, мультимодал)  ·  /kb add|ls|mkdir|find — личная база знаний (граф)[/]\n"
        f"Вложения: [dim]или упомяни путь к файлу в запросе — подхвачу автоматически[/]"
    )
    console.print(Panel(body, title="🤖 Self-Extension Agent", border_style="bright_blue", expand=False))


def render_result(result: dict) -> None:
    mode = result.get("mode", "deliberate")
    lbl, style = MODE_STYLE.get(mode, (mode.upper(), "white"))
    conf = result.get("confidence") or 0.0

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim", justify="right")
    meta.add_column()
    meta.add_row("режим", Text(lbl, style=style))
    if result.get("aim"):
        meta.add_row("цель", result["aim"])
    if result.get("standing_goal"):
        meta.add_row("🎯 стоящая", result["standing_goal"])
    if result.get("route"):
        meta.add_row("маршрут", result["route"])
    tools = (result.get("active_tools") or []) + (result.get("active_mcp_tools") or [])
    if tools:
        meta.add_row("инструменты", ", ".join(tools))
    if mode == "deliberate" or conf:
        bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        col = "green" if conf >= 0.7 else "yellow" if conf >= 0.4 else "red"
        meta.add_row("увер-сть", Text(f"{bar} {conf:.0%}", style=col))

    console.print(meta)
    answer = result.get("final_answer") or "_(пусто)_"
    console.print(Panel(Markdown(answer), title="Ответ", border_style="green", expand=True))


def cmd_facts(user_id: str) -> None:
    facts = memory_store.get_facts(user_id)
    if not facts:
        console.print("[dim]Память о пользователе пуста.[/]")
        return
    t = Table(title="Что агент знает о пользователе", border_style="cyan")
    t.add_column("ключ", style="cyan")
    t.add_column("значение")
    t.add_column("теги", style="dim")
    for f in facts:
        import json
        tags = ", ".join(json.loads(f["tags"] or "[]")) if "tags" in f.keys() else ""
        t.add_row(f["key"], f["value"], tags)
    console.print(t)


def cmd_goal(user_id: str) -> None:
    g = memory_store.get_active_goal(user_id)
    if not g:
        console.print("[dim]Активной стоящей цели нет.[/]")
        return
    crit = memory_store.goal_criteria(g)
    body = Text(g["aim"])
    if crit:
        body.append("\n\nКритерии:\n", style="dim")
        body.append("\n".join(f"  ☐ {c}" for c in crit))
    console.print(Panel(body, title="🎯 Стоящая цель", border_style="yellow", expand=False))


def cmd_diagnose(user_id: str) -> None:
    rep = diagnose(memory_store, user_id)
    style = "green" if rep["healthy"] else "red"
    console.print(Panel("\n".join(rep["findings"]) or "Проблем не найдено.",
                        title="🩺 Самодиагностика", border_style=style, expand=False))
    md = rep.get("mode_distribution") or {}
    if md.get("total"):
        console.print(f"[dim]Режимы ({md['total']} эпизодов): {md['modes']} · "
                      f"дёшево (fast/clarify): {md['cheap_share']:.0%}[/]")
    cmd_traces()


def cmd_traces() -> None:
    stats = trace_store.node_stats(24.0)
    if not stats:
        console.print("[dim]Трейсов пока нет.[/]")
        return
    t = Table(title="Трейс по нодам (24ч)", border_style="bright_black")
    for c in ("нода", "вызовов", "avg ms", "max ms", "ошибок"):
        t.add_column(c)
    for s in stats:
        t.add_row(s["node"], str(s["calls"]), f"{s['avg_ms']:.0f}", f"{s['max_ms']:.0f}", str(s["errors"]))
    console.print(t)


def _k(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def cmd_model(args: list[str]) -> None:
    if not args:
        console.print(f"Активно: [cyan]{active_summary()}[/]")
        console.print("[dim]/model api  |  /model ollama [имя_модели]  (напр. /model ollama nemotron-3-nano:4b-q8_0)[/]")
        return
    target = args[0].lower()
    if target in ("api", "openrouter", "cloud"):
        set_provider("openrouter")
    elif target == "ollama":
        set_provider("ollama", args[1] if len(args) > 1 else None)
    else:
        console.print("[red]Не понял. Используй: /model api | /model ollama [имя][/]")
        return
    with console.status("[cyan]Переключаю модель…"):
        rebuild_llms()
    console.print(f"✅ Активно → [cyan]{active_summary()}[/]")


def cmd_usage(tracker: TokenTracker) -> None:
    at = load_alltime()
    t = Table(title="🧮 Расход токенов", border_style="magenta")
    t.add_column(""); t.add_column("вход", justify="right"); t.add_column("выход", justify="right")
    t.add_column("вызовов", justify="right"); t.add_column("~$", justify="right")
    t.add_row("сессия", _k(tracker.input), _k(tracker.output), str(tracker.calls), f"${tracker.cost():.4f}")
    t.add_row("всего", _k(at['input']), _k(at['output']), str(at['calls']),
              f"${cost_of(at['input'], at['output']):.4f}")
    console.print(t)
    console.print("[dim]Оценка $ по ставкам gpt-4o-mini; модели разные — это грубо.[/]")


async def cmd_improve() -> None:
    from src.improve import graph_backward

    with console.status("[yellow]backward по графу: credit assignment + оптимизация…"):
        res = await asyncio.to_thread(graph_backward, memory_store, 3)
    console.print(Panel(str(res), title="🔧 Self-learning", border_style="magenta", expand=False))


async def _record_voice() -> str:
    """
    Голосовой ввод в CLI: запись с микрофона (ffmpeg avfoundation) до Enter,
    затем расшифровка fast-моделью. Терминалу нужен доступ к микрофону
    (System Settings → Privacy → Microphone) — macOS спросит при первом запуске.
    """
    import shutil as _shutil
    import subprocess as _sp
    import tempfile as _tmp

    if not _shutil.which("ffmpeg"):
        console.print("[red]Нужен ffmpeg: brew install ffmpeg[/]")
        return ""
    wav = Path(_tmp.mkstemp(suffix=".wav")[1])
    proc = _sp.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "avfoundation", "-i", ":0",
         "-ac", "1", "-ar", "16000", str(wav)],
        stdin=_sp.DEVNULL, stderr=_sp.PIPE,
    )
    console.print("[bold red]● Запись…[/] говори, [bold]Enter[/] — стоп")
    await asyncio.to_thread(input)
    proc.terminate()
    proc.wait(timeout=10)
    if not wav.exists() or wav.stat().st_size < 1000:
        err = (proc.stderr.read().decode(errors="ignore") if proc.stderr else "")[:200]
        console.print(f"[red]Запись пуста. Проверь доступ терминала к микрофону. {err}[/]")
        return ""
    with console.status("[cyan]Расшифровываю…"):
        text = await asyncio.to_thread(transcribe_audio, str(wav))
    wav.unlink(missing_ok=True)
    return text.strip()


def _augment_attachments(query: str) -> str:
    """
    Вложения в REPL: если в запросе упомянут существующий файл (картинка/текст/аудио),
    подкладываем его содержимое/vision-описание в контекст запроса.
    """
    known = IMAGE_EXTS | TEXT_EXTS | AUDIO_EXTS | DOC_EXTS
    paths = [
        t for t in query.replace("'", " ").replace('"', " ").split()
        if Path(t).expanduser().suffix.lower() in known and Path(t).expanduser().exists()
    ]
    if not paths:
        return query
    console.print(f"[dim]📎 вложения: {', '.join(Path(p).name for p in paths)}[/]")
    ctx = attachment_context([str(Path(p).expanduser()) for p in paths], query)
    return f"{query}\n\n=== ВЛОЖЕНИЯ ===\n{ctx}"


async def _repl_clarify(items: list[dict]) -> list[str]:
    """Онбординг неясной задачи в REPL: задаём вопросы с маркерами, пустой ответ → допущение."""
    console.print(Panel("Чтобы сделать правильно, уточни несколько деталей "
                        "([dim]Enter — оставить на моё усмотрение[/]):",
                        title="❓ Уточнение задачи", border_style="magenta", expand=False))
    answers: list[str] = []
    for i, it in enumerate(items, 1):
        q = it.get("question", "")
        opts = it.get("options") or []
        if opts:
            console.print(f"[bold]{i}. {q}[/]")
            for j, o in enumerate(opts, 1):
                console.print(f"   [cyan]{j}[/]) {o}")
            raw = (await _paused_input("   выбор (номер или свой текст): ")).strip()
            if raw.isdigit() and 1 <= int(raw) <= len(opts):
                answers.append(opts[int(raw) - 1])
            else:
                answers.append(raw)  # свой текст или пусто → допущение
        else:
            raw = (await _paused_input(f"[{i}] {q}\n   ")).strip()
            answers.append(raw)
    return answers


async def cmd_attach(args: list[str], session_id: str) -> None:
    """Приложить файл(ы) к ТЕКУЩЕЙ сессии (tmp, мультимодал — pdf/image/audio/video)."""
    paths = [Path(a).expanduser() for a in args]
    if not paths:
        console.print("[dim]/attach <путь> [ещё...] — приложить файл(ы) к этой сессии[/]"); return
    for p in paths:
        if not p.exists():
            console.print(f"[red]нет файла: {p}[/]"); continue
        msg = await asyncio.to_thread(add_session_file, session_id, str(p))
        console.print(f"[green]📎 {msg}[/]")


async def cmd_kb(args: list[str], user_id: str) -> None:
    """Глобальная база знаний: add <файл> [папка] · ls · mkdir <папка> · find <запрос>."""
    sub = (args[0].lower() if args else "")
    if sub == "add" and len(args) >= 2:
        folder = args[2] if len(args) >= 3 else ""
        console.print("[dim]индексирую в граф LightRAG…[/]")
        msg = await add_document_async(user_id, str(Path(args[1]).expanduser()), folder)
        console.print(f"[green]📚 {msg}[/]")
    elif sub == "ls":
        console.print(Panel(await asyncio.to_thread(list_kb, user_id), title="📚 База знаний", border_style="cyan", expand=False))
    elif sub == "mkdir" and len(args) >= 2:
        rel = await asyncio.to_thread(create_folder, user_id, args[1])
        console.print(f"[green]📁 создана папка: {rel}[/]")
    elif sub == "find" and len(args) >= 2:
        res = await search_kb_async(user_id, " ".join(args[1:]))
        console.print(Panel(res[:1500], title="🔎 БЗ", border_style="cyan", expand=False))
    else:
        console.print("[dim]/kb add <файл> [папка]  ·  /kb ls  ·  /kb mkdir <папка>  ·  /kb find <запрос>[/]")


async def _repl_confirm(description: str) -> bool:
    """Human-in-the-loop: подтверждение side-effect действия прямо в терминале."""
    console.print(Panel(description, title="⚠️  Агент просит разрешение на действие",
                        border_style="red", expand=False))
    ans = await _paused_input("Разрешить? [y/N] ")  # пауза спиннера, иначе ввод не пройдёт
    return ans.strip().lower() in ("y", "yes", "да", "д")


async def main():
    set_confirmer(_repl_confirm)
    set_clarifier(_repl_clarify)
    async with make_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        thread_id = str(uuid.uuid4())
        user_id = "local"
        chat_history: list[dict] = []
        banner()

        # Полноценный ввод: редактирование строки (стрелки, Ctrl+A/E, Opt+←→ по словам),
        # история ↑/↓ (постоянная между сессиями).
        Path("data").mkdir(exist_ok=True)
        session: PromptSession = PromptSession(history=FileHistory("data/.repl_history"))
        prompt_html = HTML("\n<b><ansibrightblue>›</ansibrightblue></b> ")
        tracker = TokenTracker()  # учёт токенов за сессию (через callback)

        while True:
            try:
                query = (await session.prompt_async(prompt_html)).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            low = query.lower()
            if low in ("exit", "quit", "q"):
                break
            if low in ("/new", "new"):
                clear_session(thread_id)  # ярус 3: временные файлы старой сессии не переносим
                thread_id = str(uuid.uuid4())
                chat_history = []
                console.print("[dim]Новый тред (приложенные файлы сессии очищены).[/]")
                continue
            if low.startswith("/attach"):
                await cmd_attach(query.split()[1:], thread_id); continue
            if low.startswith("/kb"):
                await cmd_kb(query.split()[1:], user_id); continue
            if low in ("/help", "help", "?"):
                banner(); continue
            if low.startswith("/model"):
                cmd_model(query.split()[1:]); continue
            if low == "/facts":
                cmd_facts(user_id); continue
            if low == "/goal":
                cmd_goal(user_id); continue
            if low == "/diagnose":
                cmd_diagnose(user_id); continue
            if low == "/traces":
                cmd_traces(); continue
            if low == "/improve":
                await cmd_improve(); continue
            if low == "/usage":
                cmd_usage(tracker); continue
            if low == "/voice":
                query = await _record_voice()
                if not query:
                    continue
                console.print(f"[bold]🎙 «{query}»[/]")
                low = query.lower()

            pre_in, pre_out, pre_calls = tracker.snapshot()
            try:
                global _live_status
                with console.status("[cyan]Думаю…", spinner="dots") as status:
                    _live_status = status  # чтобы HITL/уточнения могли поставить спиннер на паузу
                    full_query = await asyncio.to_thread(_augment_attachments, query)

                    def _show(label: str) -> None:
                        spent = tracker.total - pre_in - pre_out
                        status.update(f"[cyan]{label}[/] [dim]· 🧮 {_k(spent)} tok[/]")

                    result = await stream_with_progress(
                        graph,
                        {"query": full_query, "user_id": user_id, "session_id": thread_id,
                         "chat_history": chat_history + [{"role": "user", "content": query}]},
                        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 50,
                                "callbacks": [tracker]},
                        on_label=_show,
                    )
            except Exception as e:  # noqa: BLE001
                console.print(Panel(f"{type(e).__name__}: {e}", title="Ошибка", border_style="red"))
                continue
            finally:
                _live_status = None

            answer = result.get("final_answer", "")
            chat_history += [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
            chat_history = chat_history[-20:]
            render_result(result)

            # Расход токенов за этот запрос + персист all-time.
            di, do = tracker.input - pre_in, tracker.output - pre_out
            add_alltime(di, do, tracker.calls - pre_calls)
            console.print(f"[dim]🧮 токены: {_k(di)} in + {_k(do)} out = {_k(di+do)} "
                          f"(~${cost_of(di, do):.4f}) · сессия {_k(tracker.total)} · /usage[/]")

        clear_session(thread_id)  # ярус 3 не переживает выход из сессии
        console.print("[dim]Пока![/]")


if __name__ == "__main__":
    asyncio.run(main())
