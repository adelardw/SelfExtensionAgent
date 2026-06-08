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

from src.agent import build_graph, config, memory_store
from src.tracing import diagnose, trace_store
from src.usage import TokenTracker, add_alltime, cost_of, load_alltime

console = Console()

# Цвета режимов мышления (Self-Reflexion Choice).
MODE_STYLE = {
    "fast": ("⚡ FAST", "bold green"),
    "reason": ("🧠 REASON", "bold cyan"),
    "deliberate": ("🛠 DELIBERATE", "bold yellow"),
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
        f"Модель: [cyan]{config.model.name}[/] · код: [cyan]{config.code_model.name}[/]\n"
        f"Режимы: {legend}\n"
        f"Команды: [dim]/help /new /facts /goal /diagnose /traces /improve /usage  ·  exit[/]"
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


async def main():
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
                thread_id = str(uuid.uuid4())
                chat_history = []
                console.print("[dim]Новый тред.[/]")
                continue
            if low in ("/help", "help", "?"):
                banner(); continue
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

            pre_in, pre_out, pre_calls = tracker.snapshot()
            try:
                with console.status("[cyan]Думаю…", spinner="dots"):
                    result = await graph.ainvoke(
                        {"query": query, "user_id": user_id,
                         "chat_history": chat_history + [{"role": "user", "content": query}]},
                        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 50,
                                "callbacks": [tracker]},
                    )
            except Exception as e:  # noqa: BLE001
                console.print(Panel(f"{type(e).__name__}: {e}", title="Ошибка", border_style="red"))
                continue

            answer = result.get("final_answer", "")
            chat_history += [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
            chat_history = chat_history[-20:]
            render_result(result)

            # Расход токенов за этот запрос + персист all-time.
            di, do = tracker.input - pre_in, tracker.output - pre_out
            add_alltime(di, do, tracker.calls - pre_calls)
            console.print(f"[dim]🧮 токены: {_k(di)} in + {_k(do)} out = {_k(di+do)} "
                          f"(~${cost_of(di, do):.4f}) · сессия {_k(tracker.total)} · /usage[/]")

        console.print("[dim]Пока![/]")


if __name__ == "__main__":
    asyncio.run(main())
