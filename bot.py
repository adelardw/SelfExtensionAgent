import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
warnings.filterwarnings("ignore", message="urllib3")

import asyncio
import json
import logging
import os
import sqlite3
import uuid

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters.command import Command
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import build_graph, memory_store
from src.tracing import diagnose

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not API_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
agent_app = build_graph(checkpointer=SqliteSaver(conn))

MODE_EMOJI = {"fast": "⚡ fast", "reason": "🧠 reason", "deliberate": "🛠 deliberate", "clarify": "❓ clarify"}
_threads: dict[str, str] = {}  # user_id → thread_id (для /new)


def _thread(user_id: str) -> str:
    return _threads.setdefault(user_id, user_id)


async def _send_long(message: types.Message, text: str) -> None:
    for i in range(0, len(text), 4000):
        await message.answer(text[i:i + 4000])


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я саморасширяющийся ассистент с памятью, навыками и доступом к устройству.\n"
        "Просто пиши задачу. Команды: /facts /goal /diagnose /new"
    )


@dp.message(Command("new"))
async def cmd_new(message: types.Message):
    _threads[str(message.from_user.id)] = uuid.uuid4().hex
    await message.answer("🔄 Новый контекст диалога.")


@dp.message(Command("facts"))
async def cmd_facts(message: types.Message):
    uid = str(message.from_user.id)
    facts = memory_store.get_facts(uid)
    if not facts:
        await message.answer("Пока ничего о тебе не запомнил.")
        return
    lines = [f"• {f['key']}: {f['value']}" for f in facts[:20]]
    await message.answer("🧠 Что я знаю о тебе:\n" + "\n".join(lines))


@dp.message(Command("goal"))
async def cmd_goal(message: types.Message):
    g = memory_store.get_active_goal(str(message.from_user.id))
    if not g:
        await message.answer("Активной цели нет.")
        return
    crit = memory_store.goal_criteria(g)
    txt = f"🎯 Цель: {g['aim']}"
    if crit:
        txt += "\nКритерии:\n" + "\n".join(f"  ☐ {c}" for c in crit)
    await message.answer(txt)


@dp.message(Command("diagnose"))
async def cmd_diagnose(message: types.Message):
    rep = diagnose(memory_store, str(message.from_user.id))
    body = "✅ Проблем не найдено." if rep["healthy"] else "⚠ Найдено:\n" + "\n".join(f"• {f}" for f in rep["findings"])
    await message.answer("🩺 Самодиагностика:\n" + body)


@dp.message(F.text)
async def handle_message(message: types.Message):
    uid = str(message.from_user.id)
    status = await message.answer("🤔 Думаю…")
    try:
        cfg = {"configurable": {"thread_id": _thread(uid)}, "recursion_limit": 50}
        result = await agent_app.ainvoke(
            {"query": message.text, "user_id": uid, "chat_history": []}, config=cfg
        )
        answer = result.get("final_answer") or "Не смог сформировать ответ."

        mode = MODE_EMOJI.get(result.get("mode", ""), result.get("mode", ""))
        conf = result.get("confidence") or 0.0
        tools = (result.get("active_tools") or []) + (result.get("active_mcp_tools") or [])
        head = mode + (f" · {conf:.0%}" if conf else "")
        if result.get("aim"):
            head += f"\n🎯 {result['aim']}"
        if tools:
            head += f"\n🔧 {', '.join(tools)}"

        await status.delete()
        await message.answer(head)
        await _send_long(message, answer)
    except Exception as e:
        logger.error(f"agent error: {e}")
        await status.edit_text(f"⚠ Ошибка: {e}")


async def main():
    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
