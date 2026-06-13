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
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

console = Console()
# Мгновенный фидбек: тяжёлые импорты (langchain/langgraph/модели/навыки) занимают пару
# секунд — показываем это сразу, чтобы старт не выглядел зависшим.
with console.status("[cyan]🔥 Прогрев агента (первый запуск дольше — грузятся модели и навыки)…"):
    from src.agent import build_graph, config, memory_store, rebuild_llms
    from src.clarify import set_clarifier
    from src.hitl import set_confirmer
    from src.llm import active_summary, set_provider
    from src.media import (AUDIO_EXTS, DOC_EXTS, IMAGE_EXTS, TEXT_EXTS,
                           attachment_context, transcribe_audio)
    from src.knowledge_base import (add_document_async, add_session_file, create_folder,
                                    list_kb, clear_session, search_kb_async)
    from src.progress import stream_with_progress
    from src.tracing import diagnose, trace_store
    from src.usage import TokenTracker, add_alltime, cost_of, load_alltime

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
WORK_LABEL = {
    "manual": "✋ ручной (подтверждать действия)",
    "auto-accept": "🟡 auto-accept (действия подтверждаются автоматически)",
    "auto": "⚡ auto (агент автономен: сам мышление, развилки, действия)",
}

# Слэш-команды с описаниями: автодополнение при вводе «/» (как в qwen-code/claude-code CLI).
COMMANDS = {
    "/config":   "настройки: модель · режим работы · мышление · гранты",
    "/auto":     "режим работы: /auto — полный авто · /auto accept · /auto off",
    "/model":    "модель: /model api · /model ollama [имя]",
    "/backend":  "браузер-движок: /backend puppeteer|hybrid|extension (puppeteer ждёт SPA)",
    "/voice":    "голосовой ввод (один запрос)",
    "/attach":   "приложить файл к сессии (pdf/image/audio/video)",
    "/kb":       "база знаний: add <файл> · ls · mkdir <папка> · find <запрос>",
    "/facts":    "что агент помнит о тебе",
    "/goal":     "текущая стоящая цель",
    "/usage":    "расход токенов и $",
    "/diagnose": "самодиагностика",
    "/traces":   "трейс по нодам",
    "/improve":  "самообучение вручную",
    "/new":      "новый тред (вложения сессии очищаются)",
    "/help":     "показать баннер",
    "exit":      "выход (также quit · q · Ctrl+D)",
}


class SlashCompleter(Completer):
    """Дропдаун команд с описаниями при вводе «/» (и «exit» по первым буквам)."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip().lower()
        if not text or not (text.startswith("/") or "exit".startswith(text)):
            return
        for cmd, desc in COMMANDS.items():
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display=cmd, display_meta=desc)


MODE_STYLE = {
    "fast": ("⚡ FAST", "bold green"),
    "reason": ("🧠 REASON", "bold cyan"),
    "act": ("🤚 ACT", "bold blue"),
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
    from src import hitl
    from src.cli_config import get_cli

    legend = "  ".join(f"[{st}]{lbl}[/]" for lbl, st in MODE_STYLE.values())
    work = WORK_LABEL.get(hitl.work_mode(), hitl.work_mode())
    force = (get_cli("force_mode") or "").strip()
    if force:
        work += f"  ·  мышление: {force}"
    body = Text.from_markup(
        f"Активно: [cyan]{active_summary()}[/]\n"
        f"Режим работы: [bold]{work}[/]  [dim](/config — настроить, /auto — переключить)[/]\n"
        f"Режимы мышления: {legend}\n"
        f"Физический веб: [dim]просто скажи словами — «включи музыку <A>», «открой моё избранное», "
        f"«поставь паузу», «найди фильм <X>», «закажи <еду>» — агент сам выберет сервис и сделает[/]\n"
        f"Команды: [dim]/config /model /auto /voice /help /new /facts /goal /diagnose /traces /improve /usage  ·  exit[/]\n"
        f"Подтверждения: [dim]отвечай словами — «да» · «да, всегда» (больше не спрашивать) · «да, но …» · «нет, …» · или скажи, как сделать иначе[/]\n"
        f"Файлы: [dim]/attach <файл> — в сессию (tmp, мультимодал)  ·  /kb add|ls|mkdir|find — личная база знаний (граф)[/]\n"
        f"Автозапуск: [dim]uv run main.py \"задача\" [--auto] — one-shot без REPL (для скриптов/cron)[/]"
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
    from src.cli_config import set_cli

    if not args:
        console.print(f"Активно: [cyan]{active_summary()}[/]")
        console.print("[dim]/model api  |  /model ollama [имя_модели]  (напр. /model ollama nemotron-3-nano:4b-q8_0)[/]")
        return
    target = args[0].lower()
    if target in ("api", "openrouter", "cloud"):
        set_provider("openrouter")
        set_cli("provider", "openrouter"); set_cli("model", None)
    elif target == "ollama":
        model = args[1] if len(args) > 1 else None
        set_provider("ollama", model)
        set_cli("provider", "ollama"); set_cli("model", model)
    else:
        console.print("[red]Не понял. Используй: /model api | /model ollama [имя][/]")
        return
    with console.status("[cyan]Переключаю модель…"):
        rebuild_llms()
    console.print(f"✅ Активно → [cyan]{active_summary()}[/]  [dim](сохранено в config.local.yml)[/]")


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


def _resolve_choice(raw: str, opts: list[str]) -> str:
    """Ответ на вопрос с вариантами: номер → вариант; префикс текста варианта → вариант;
    «сам реши/не знаю/любой» → допущение (''); иначе — свой текст как есть."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.isdigit() and 1 <= int(raw) <= len(opts):
        return opts[int(raw) - 1]
    low = raw.lower()
    if low in ("сам", "сам реши", "не знаю", "любой", "без разницы", "на твое усмотрение",
               "на твоё усмотрение", "skip", "пропусти"):
        return ""
    for o in opts:  # «яндекс» матчит вариант «Яндекс Музыка»
        if o.lower().startswith(low) or low in o.lower():
            return o
    return raw


async def _ask_one(i: int, n: int, it: dict) -> str:
    """Один вопрос опросника: варианты + явный плейсхолдер, что и как можно ответить."""
    q = it.get("question", "")
    why = it.get("why", "")
    opts = it.get("options") or []
    body = f"[bold]{q}[/]"
    if why:
        body += f"\n[dim]{why}[/]"
    if opts:
        body += "\n" + "\n".join(f"  [cyan]{j}[/]) {o}" for j, o in enumerate(opts, 1))
        ph = f"номер 1-{len(opts)} · свой текст · Enter = на моё усмотрение"
    else:
        ph = "свободный ответ · Enter = на моё усмотрение"
    console.print(Panel(body, title=f"❓ {i}/{n}", subtitle=f"[dim]{ph}[/]",
                        border_style="magenta", expand=False))
    raw = (await _paused_input("   › ")).strip()
    return _resolve_choice(raw, opts) if opts else raw


async def _repl_clarify(items: list[dict]) -> list[str]:
    """Опросник (Q/A-секции, как форма): вопрос-за-вопросом с плейсхолдерами, в конце —
    резюме ответов и САБМИТ (Enter — отправить, номер — поправить ответ).
    В auto-режиме агент автономен целиком: вопросы не задаются, развилки решает сам
    (разумные допущения, в финале пометит «исходил из того, что…»)."""
    from src import hitl
    if hitl.full_auto():
        console.print("[dim]⚡ auto: уточнения не задаю — беру разумные допущения[/]")
        return []
    n = len(items)
    answers: list[str] = []
    for i, it in enumerate(items, 1):
        answers.append(await _ask_one(i, n, it))
    if n > 1:  # резюме + сабмит только когда есть что резюмировать
        while True:
            lines = []
            for i, (it, a) in enumerate(zip(items, answers), 1):
                shown = a or "[dim](на усмотрение агента)[/]"
                lines.append(f"[cyan]{i}[/]. {it.get('question', '')[:70]}\n   → {shown}")
            console.print(Panel("\n".join(lines), title="📋 Твои ответы",
                                subtitle="[dim]Enter — отправить · номер — изменить ответ[/]",
                                border_style="magenta", expand=False))
            raw = (await _paused_input("   › ")).strip()
            if not raw:
                break
            if raw.isdigit() and 1 <= int(raw) <= n:
                idx = int(raw) - 1
                answers[idx] = await _ask_one(idx + 1, n, items[idx])
            else:
                break  # любой другой ввод — отправляем как есть
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
        path = str(Path(args[1]).expanduser())
        # Граф стоит денег (LLM-извлечение сущностей на каждый чанк) — прикидываем цену и
        # спрашиваем ДО запуска (HITL). Отказ ≠ отмена: документ ляжет в бесплатный BM25.
        use_graph = False
        from src.lightrag_engine import estimate_index_cost, lightrag_available
        if lightrag_available():
            from src.media import read_file
            text = await asyncio.to_thread(read_file, path, 200_000)
            est = estimate_index_cost(text)
            console.print(f"[yellow]⚖ граф LightRAG: ~{est['chunks']} чанков, "
                          f"~{est['calls']} LLM-вызовов, ≈ ${est['usd']:.3f}[/]")
            ans = await _paused_input("Индексировать в граф? (да / нет — тогда только бесплатный BM25): ")
            from src.semantics import parse_assent
            use_graph = parse_assent(ans)[0] is True
        if use_graph:
            console.print("[dim]индексирую в граф LightRAG…[/]")
        msg = await add_document_async(user_id, path, folder, use_graph=use_graph)
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


_FORCE_MODES = ("fast", "reason", "act", "deliberate", "heavy")


async def cmd_config() -> None:
    """Панель настроек CLI: видно текущий режим работы, меняется по номеру, персистится
    в config.local.yml (мердж поверх config.yml) и применяется сразу."""
    from src import hitl
    from src.cli_config import get_cli, set_cli

    while True:
        force = (get_cli("force_mode") or "").strip() or "auto"
        force_lbl = "авто — агент решает сам" if force == "auto" else MODE_STYLE.get(force, (force,))[0]
        allow = list(get_cli("allow") or [])
        rows = [
            f"[cyan]1[/]. Модель: [bold]{active_summary()}[/]",
            f"[cyan]2[/]. Режим работы: [bold]{WORK_LABEL.get(hitl.work_mode(), hitl.work_mode())}[/]",
            f"[cyan]3[/]. Режим мышления: [bold]{force_lbl}[/]",
            f"[cyan]4[/]. Разрешено без вопроса: [bold]{len(allow)}[/] " +
            (f"[dim]({', '.join(allow[:3])}{'…' if len(allow) > 3 else ''})[/]" if allow else "[dim](пусто)[/]"),
        ]
        console.print(Panel("\n".join(rows), title="⚙️  Настройки",
                            subtitle="[dim]номер — изменить · Enter или q — закрыть[/]",
                            border_style="bright_blue", expand=False))
        pick = (await _paused_input("   › ")).strip().lower()
        if not pick or pick in ("q", "й", "exit", "выход", "закрыть"):
            return
        if pick == "1":
            raw = (await _paused_input(
                "   модель (api · ollama [имя] · Enter — назад) › ")).strip()
            if raw:
                cmd_model(raw.split())
        elif pick == "2":
            modes = list(hitl.WORK_MODES)
            console.print("\n".join(f"   [cyan]{i}[/]) {WORK_LABEL[m]}" for i, m in enumerate(modes, 1)))
            raw = (await _paused_input(f"   режим (1-{len(modes)} · Enter — назад) › ")).strip()
            if raw.isdigit() and 1 <= int(raw) <= len(modes):
                m = hitl.set_work_mode(modes[int(raw) - 1])
                set_cli("work_mode", m)
                console.print(f"   {WORK_LABEL[m]} [dim](сохранено)[/]")
        elif pick == "3":
            opts = ["авто — агент решает сам (рекомендуется)"] + [MODE_STYLE[m][0] for m in _FORCE_MODES]
            console.print("\n".join(f"   [cyan]{i}[/]) {o}" for i, o in enumerate(opts, 1)))
            raw = (await _paused_input(f"   режим (1-{len(opts)} · Enter — назад) › ")).strip()
            if raw.isdigit() and 1 <= int(raw) <= len(opts):
                mode = "" if int(raw) == 1 else _FORCE_MODES[int(raw) - 2]
                set_cli("force_mode", mode)
                console.print(f"   режим: {mode or 'auto'} [dim](сохранено)[/]")
        elif pick == "4":
            raw = (await _paused_input("   очистить все гранты? (да/нет) › ")).strip()
            from src.semantics import parse_assent
            if parse_assent(raw)[0] is True:
                set_cli("allow", [])
                hitl._grants.clear()
                console.print("   гранты очищены")


async def _repl_confirm(description: str) -> str:
    """Human-in-the-loop: СЕМАНТИЧЕСКОЕ подтверждение — отвечай словами, не [y/N].
    Возвращаем сырой текст; hitl.confirm_rich разберёт: «да» / «да, но …» (условие —
    агент скорректирует) / «нет, потому что …» / своё указание (агент последует ему)."""
    console.print(Panel(description, title="⚠️  Агент просит разрешение на действие",
                        subtitle="[dim]да · «да, всегда» · «да, но …» · нет · или скажи, как сделать иначе[/]",
                        border_style="red", expand=False))
    return await _paused_input("   › ")  # пауза спиннера, иначе ввод не пройдёт


async def main():
    set_confirmer(_repl_confirm)
    set_clarifier(_repl_clarify)
    # CLI-настройки из config.local.yml (мердж поверх config.yml): модель, auto-режим,
    # гранты «да, всегда» — всё, что менялось из CLI, применяется автоматически.
    from src import hitl
    from src.cli_config import get_cli
    prov = get_cli("provider")
    if prov == "ollama":
        set_provider("ollama", get_cli("model"))
        rebuild_llms()
        console.print(f"[dim]из config.local.yml: {active_summary()}[/]")
    hitl.load_grants(get_cli("allow") or [])
    wm = get_cli("work_mode") or ("auto-accept" if get_cli("auto_confirm") else "manual")
    if hitl.set_work_mode(wm) != "manual":
        console.print(f"[dim]{WORK_LABEL[hitl.work_mode()]} (/auto off — вернуть ручной)[/]")
    # Мост к браузерному расширению (агент в ТВОЁМ браузере). Токен — для разовой
    # настройки расширения (side panel).
    try:
        from src import browser_bridge
        browser_bridge.ensure_server()
        await asyncio.sleep(0.4)  # дать серверу привязать порт перед сообщением
        if browser_bridge._serving:
            console.print(f"[dim]🧩 Мост браузера слушает 127.0.0.1:{browser_bridge.PORT} "
                          f"(расширение подключится автоматически).[/]")
        if not get_cli("browser_bridge_seen"):
            console.print(f"[dim]   Расширение (физический веб + чат из браузера): "
                          f"chrome://extensions → Developer mode → Load unpacked → папка "
                          f"extension/ ; токен в боковую панель:[/] [cyan]{browser_bridge.token()}[/]")
            from src.cli_config import set_cli as _sc
            _sc("browser_bridge_seen", True)
    except Exception as e:  # noqa: BLE001
        print(f"[bridge] не поднялся: {e}")
    async with make_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        thread_id = str(uuid.uuid4())
        user_id = "local"
        chat_history: list[dict] = []
        banner()

        # Чат из браузерного расширения (side panel) → тот же граф, своя ветка истории.
        ext_thread = str(uuid.uuid4())
        ext_history: list[dict] = []  # НАКОПИТЕЛЬНАЯ история панели (как в REPL) — контекст не теряется

        async def _ext_chat(text: str) -> str:
            from src.cli_config import get_cli as _gc
            tracker = TokenTracker()  # расход токенов чата расширения — виден в панели И в CLI
            res = await graph.ainvoke(
                {"query": text, "user_id": user_id, "session_id": ext_thread,
                 "force_mode": ("" if hitl.full_auto() else (_gc("force_mode") or "")),
                 "chat_history": ext_history + [{"role": "user", "content": text}]},
                config={"configurable": {"thread_id": ext_thread}, "recursion_limit": 50,
                        "callbacks": [tracker]},
            )
            ans = res.get("final_answer", "") or "(пустой ответ)"
            ext_history.append({"role": "user", "content": text})
            ext_history.append({"role": "assistant", "content": ans})
            del ext_history[:-20]  # держим последние 20 реплик
            di, do = tracker.input, tracker.output
            # Дублируем расход в терминал агента (чтобы и там было видно, что делает панель).
            console.print(f"[dim]🧩 чат-расширение: «{text[:40]}» · {_k(di+do)} tok "
                          f"(~${cost_of(di, do):.4f})[/]")
            return f"{ans}\n\n🧮 {_k(di + do)} tok (~${cost_of(di, do):.4f})"

        try:
            browser_bridge.set_chat_handler(_ext_chat)
        except Exception:  # noqa: BLE001
            pass

        # Полноценный ввод: редактирование строки (стрелки, Ctrl+A/E, Opt+←→ по словам),
        # история ↑/↓ (постоянная между сессиями).
        Path("data").mkdir(exist_ok=True)
        # Без complete_while_typing и bottom_toolbar: они оставляли терминал в raw-режиме
        # (Enter эхо-ился как ^M/«M») и давали подвисания. Автодополнение команд — по Tab,
        # плюс серый плейсхолдер-подсказка. Надёжно.
        session: PromptSession = PromptSession(
            history=FileHistory("data/.repl_history"),
            completer=SlashCompleter(),
            complete_in_thread=True,
            placeholder=HTML('<style fg="#666666">задача · «/»+Tab — команды · exit — выход</style>'),
        )
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
            if low in ("/config", "/settings", "config"):
                await cmd_config(); continue
            if low.startswith("/auto"):
                from src import hitl
                from src.cli_config import set_cli
                arg = (query.split()[1:] or ["auto"])[0].lower()
                m = {"off": "manual", "0": "manual", "выкл": "manual",
                     "accept": "auto-accept", "on": "auto", "auto": "auto"}.get(arg, "auto")
                hitl.set_work_mode(m); set_cli("work_mode", m)
                console.print(f"{WORK_LABEL[m]} [dim](сохранено · /auto — полный auto, /auto accept — только подтверждения, /auto off — ручной)[/]")
                continue
            if low.startswith("/attach"):
                await cmd_attach(query.split()[1:], thread_id); continue
            if low.startswith("/kb"):
                await cmd_kb(query.split()[1:], user_id); continue
            if low in ("/help", "help", "?"):
                banner(); continue
            if low.startswith("/model"):
                cmd_model(query.split()[1:]); continue
            if low.startswith("/backend"):
                from src.cli_config import get_cli, set_cli
                arg = (query.split()[1:] or [""])[0].lower()
                if arg in ("extension", "puppeteer", "hybrid", "window"):
                    set_cli("browser_backend", arg)
                    console.print(f"[green]🧩 браузер-бэкенд → [bold]{arg}[/][/] [dim](сохранено)[/]"
                                  + ("\n[dim]puppeteer/hybrid: open/see ждут рендер тяжёлого SPA "
                                     "(Я.Еда/Лавка) — нужен Reload расширения до версии с pp-бэкендом.[/]"
                                     if arg in ("puppeteer", "hybrid") else ""))
                else:
                    cur = get_cli("browser_backend") or "extension"
                    console.print(f"[cyan]браузер-бэкенд: {cur}[/]  [dim]/backend extension|puppeteer|hybrid|window"
                                  "  ·  puppeteer/hybrid = Puppeteer-ожидание SPA[/]")
                continue
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

                    from src.cli_config import get_cli as _get_cli
                    # В auto агент автономен ЦЕЛИКОМ: тип мышления выбирает сам (фиксация
                    # из /config действует только в ручном режиме).
                    _force = "" if hitl.full_auto() else (_get_cli("force_mode") or "")
                    result = await stream_with_progress(
                        graph,
                        {"query": full_query, "user_id": user_id, "session_id": thread_id,
                         "force_mode": _force,
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

            # Пост-обработка под защитой: сбой рендера/записи НЕ должен ронять весь CLI.
            try:
                answer = result.get("final_answer", "")
                chat_history += [{"role": "user", "content": query},
                                 {"role": "assistant", "content": answer}]
                chat_history = chat_history[-20:]
                render_result(result)
                di, do = tracker.input - pre_in, tracker.output - pre_out
                add_alltime(di, do, tracker.calls - pre_calls)
                console.print(f"[dim]🧮 токены: {_k(di)} in + {_k(do)} out = {_k(di+do)} "
                              f"(~${cost_of(di, do):.4f}) · сессия {_k(tracker.total)} · /usage[/]")
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]пост-обработка: {type(e).__name__}: {e}[/]")

        clear_session(thread_id)  # ярус 3 не переживает выход из сессии
        console.print("[dim]Пока![/]")


async def run_once(task: str, auto: bool = False) -> int:
    """Автоматизированный запуск (скрипты/cron/другие приложения): один запрос без REPL.
    Уточнения → разумные допущения (clarifier не регистрируется); подтверждения — по
    конфигу/грантам, --auto включает auto-accept на этот запуск. Ответ — в stdout,
    exit code 0/1 — решена ли задача (для пайплайнов)."""
    from src import hitl
    from src.cli_config import get_cli

    hitl.load_grants(get_cli("allow") or [])
    # Мост браузера поднимаем и в автоматизированном пути: расширение успевает подключиться за
    # время прогона, и агент может (а) играть музыку/видео, (б) ФОНОВО открыть итоговую ссылку
    # после анализа (критерий «сам открыть наиболее подходящую вкладку»). Идемпотентно.
    try:
        from src import browser_bridge
        browser_bridge.ensure_server()
    except Exception:  # noqa: BLE001
        pass
    if auto:
        hitl.set_work_mode("auto")  # автоматизированный запуск = полная автономия
    else:
        hitl.set_work_mode(get_cli("work_mode") or ("auto-accept" if get_cli("auto_confirm") else "manual"))
    prov = get_cli("provider")
    if prov == "ollama":
        set_provider("ollama", get_cli("model"))
        rebuild_llms()
    async with make_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        sid = str(uuid.uuid4())
        try:
            result = await graph.ainvoke(
                {"query": task, "user_id": "local", "session_id": sid,
                 # в auto агент сам выбирает тип мышления; фиксация — только ручной режим
                 "force_mode": ("" if hitl.full_auto() else (get_cli("force_mode") or "")),
                 "chat_history": [{"role": "user", "content": task}]},
                config={"configurable": {"thread_id": sid}, "recursion_limit": 50},
            )
        finally:
            clear_session(sid)
    answer = result.get("final_answer", "")
    print(answer or "(пустой ответ)")
    ok = bool(answer) and not result.get("user_blocked") and result.get("validation_passed", True)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys as _sys

    _args = [a for a in _sys.argv[1:]]
    _auto = "--auto" in _args
    _task = " ".join(a for a in _args if a != "--auto").strip()
    if _task:
        raise SystemExit(asyncio.run(run_once(_task, auto=_auto)))
    asyncio.run(main())
