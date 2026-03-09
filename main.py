import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from omegaconf import OmegaConf

from src.agent import build_graph, config


# ── Checkpointer factory ──────────────────────────────

@asynccontextmanager
async def make_checkpointer():
    """Создаёт checkpointer по config.yml и управляет его lifecycle."""
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


# ── REPL ───────────────────────────────────────────────

async def main():
    async with make_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        thread_id = str(uuid.uuid4())
        chat_history: list[dict] = []

        backend = config.get("checkpointer", {}).get("backend", "memory")
        print("Self-Extension Agent")
        print("=" * 44)
        print(f"Checkpointer: {backend}")
        print(f"Thread: {thread_id[:8]}...")
        print("Type 'exit' to quit, 'new' for new thread.\n")

        while True:
            query = input("> ").strip()
            if query.lower() in ("exit", "quit", "q", ""):
                break
            if query.lower() == "new":
                thread_id = str(uuid.uuid4())
                chat_history = []
                print(f"New thread: {thread_id[:8]}...\n")
                continue

            result = await graph.ainvoke(
                {
                    "query":                    query,
                    "messages":                 [],
                    "chat_history":             chat_history + [{"role": "user", "content": query}],
                    "route":                    "",
                    "created_skill_name":       "",
                    "create_validation_passed": False,
                    "create_feedback":          "",
                    "create_retries":           0,
                    "selected_skills":          [],
                    "plan":                     "",
                    "skill_context":            "",
                    "skill_prompts":            "",
                    "final_answer":             "",
                    "confidence":               0.0,
                    "validation_passed":        False,
                    "validation_feedback":      "",
                    "global_retries":           0,
                },
                config={"configurable": {"thread_id": thread_id}},
            )

            answer = result.get("final_answer", "No answer")

            # Обновляем chat_history ПОСЛЕ успешного вызова
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})

            # Ограничиваем историю — последние 20 сообщений (10 пар)
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

            print(f"\n{'=' * 44}")
            print(f"Answer:\n{answer}")

            fb = result.get("validation_feedback")
            if fb:
                conf = result.get("confidence", 0)
                print(f"\n[SGR] {fb}")
                print(f"[Confidence] {conf:.0%}")

            print()


if __name__ == "__main__":
    asyncio.run(main())
