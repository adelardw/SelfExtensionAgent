import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
warnings.filterwarnings("ignore", message="urllib3")

import asyncio
import contextvars
import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters.command import Command
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from src import runbudget, run_context
from src.agent import build_graph, memory_store
from src.clarify import set_clarifier
from src.hitl import set_confirmer
from src.media import attachment_context, transcribe_audio
from src.progress import stream_with_progress
from src.tracing import diagnose
from src.usage import TokenTracker, add_alltime, cost_of

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not API_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Allowlist chat_id: бот = РУКИ агента (тулзы/действия). Без allowlist любой, кто нашёл бота, слал бы
# команды; при глобальном auto-режиме — чужие руки с авто-исполнением (баг ревью). TELEGRAM_ALLOWED_IDS
# — запятые. Пусто → бот открыт всем, но с ГРОМКИМ предупреждением при старте (личный бот не ломаем).
_ALLOWED_IDS = {x.strip() for x in (os.getenv("TELEGRAM_ALLOWED_IDS") or "").split(",") if x.strip()}
if not _ALLOWED_IDS:
    logger.warning("⚠ TELEGRAM_ALLOWED_IDS не задан — бот отвечает ЛЮБОМУ Telegram-юзеру. "
                   "Для приватного бота задай TELEGRAM_ALLOWED_IDS=<твой_id>[,<id>…]")


def _authorized(uid) -> bool:
    return (not _ALLOWED_IDS) or (str(uid) in _ALLOWED_IDS)


@dp.message.outer_middleware()
async def _auth_mw(handler, event, data):
    """Единая точка: неавторизованный chat_id не доходит до агента/тулов."""
    uid = getattr(getattr(event, "from_user", None), "id", None)
    if uid is not None and not _authorized(uid):
        try:
            await event.answer("⛔ Доступ ограничен. Обратись к владельцу бота.")
        except Exception:  # noqa: BLE001
            pass
        return None
    return await handler(event, data)

conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
agent_app = build_graph(checkpointer=SqliteSaver(conn))

MODE_EMOJI = {"fast": "⚡ fast", "reason": "🧠 reason", "deliberate": "🛠 deliberate",
              "heavy": "🏗 heavy", "clarify": "❓ clarify"}
_threads: dict[str, str] = {}  # user_id → thread_id (для /new)

# ── Human-in-the-loop: подтверждение side-effect действий inline-кнопками ──
_current_msg: contextvars.ContextVar[types.Message | None] = contextvars.ContextVar("hitl_msg", default=None)
_pending_confirms: dict[str, asyncio.Future] = {}
HITL_TIMEOUT_S = 120


async def _bot_confirm(description: str) -> bool:
    """Шлёт в текущий чат запрос с кнопками; нет ответа за таймаут → отказ."""
    message = _current_msg.get()
    if message is None:
        return False
    cid = uuid.uuid4().hex[:12]
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    _pending_confirms[cid] = fut
    kb = types.InlineKeyboardMarkup(inline_keyboard=[[
        types.InlineKeyboardButton(text="✅ Разрешить", callback_data=f"hitl:{cid}:y"),
        types.InlineKeyboardButton(text="⛔ Отклонить", callback_data=f"hitl:{cid}:n"),
    ]])
    await message.answer(f"⚠️ Агент просит разрешение на действие:\n`{description}`",
                         reply_markup=kb, parse_mode="Markdown")
    try:
        return await asyncio.wait_for(fut, timeout=HITL_TIMEOUT_S)
    except asyncio.TimeoutError:
        return False
    finally:
        _pending_confirms.pop(cid, None)


@dp.callback_query(F.data.startswith("hitl:"))
async def hitl_callback(cb: types.CallbackQuery):
    _, cid, ans = cb.data.split(":", 2)
    fut = _pending_confirms.get(cid)
    if fut is not None and not fut.done():
        fut.set_result(ans == "y")
    await cb.answer("Разрешено" if ans == "y" else "Отклонено")
    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:  # noqa: BLE001
        pass


set_confirmer(_bot_confirm)


# ── Уточнения (онбординг неясной задачи): варианты-кнопки или ответ текстом ──
_pending_opts: dict[str, asyncio.Future] = {}      # cid → выбор варианта
_pending_text: dict[str, asyncio.Future] = {}      # uid → ответ свободным текстом
CLARIFY_TIMEOUT_S = 180


async def _bot_clarify(items: list[dict]) -> list[str]:
    message = _current_msg.get()
    if message is None:
        return []
    uid = str(message.from_user.id)
    await message.answer("❓ Чтобы сделать правильно, уточни пару деталей "
                         "(не ответишь — решу сам разумным образом):")
    answers: list[str] = []
    for it in items:
        q = it.get("question", "")
        opts = it.get("options") or []
        if opts:
            cid = uuid.uuid4().hex[:12]
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            _pending_opts[cid] = fut
            rows = [[types.InlineKeyboardButton(text=o[:60], callback_data=f"clr:{cid}:{j}")]
                    for j, o in enumerate(opts)]
            rows.append([types.InlineKeyboardButton(text="🤖 на твоё усмотрение", callback_data=f"clr:{cid}:-1")])
            await message.answer(f"❓ {q}", reply_markup=types.InlineKeyboardMarkup(inline_keyboard=rows))
            try:
                idx = await asyncio.wait_for(fut, timeout=CLARIFY_TIMEOUT_S)
                answers.append(opts[idx] if 0 <= idx < len(opts) else "")
            except asyncio.TimeoutError:
                answers.append("")
            finally:
                _pending_opts.pop(cid, None)
        else:
            fut = asyncio.get_running_loop().create_future()
            _pending_text[uid] = fut
            await message.answer(f"❓ {q}\n(ответь сообщением или подожди — решу сам)")
            try:
                answers.append(await asyncio.wait_for(fut, timeout=CLARIFY_TIMEOUT_S))
            except asyncio.TimeoutError:
                answers.append("")
            finally:
                _pending_text.pop(uid, None)
    return answers


@dp.callback_query(F.data.startswith("clr:"))
async def clarify_callback(cb: types.CallbackQuery):
    _, cid, idx = cb.data.split(":", 2)
    fut = _pending_opts.get(cid)
    if fut is not None and not fut.done():
        fut.set_result(int(idx))
    await cb.answer("Принято")
    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:  # noqa: BLE001
        pass


set_clarifier(_bot_clarify)


def _thread(user_id: str) -> str:
    return _threads.setdefault(user_id, user_id)


async def _send_long(message: types.Message, text: str) -> None:
    for i in range(0, len(text), 4000):
        await message.answer(text[i:i + 4000])


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я саморасширяющийся ассистент с памятью, навыками и доступом к устройству.\n"
        "Пиши задачу текстом, голосом 🎙, кидай картинки 🖼 и файлы 📎.\n"
        "Команды: /facts /goal /diagnose /new"
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


UPLOADS = Path("data/uploads")
UPLOADS.mkdir(parents=True, exist_ok=True)


async def _download(file_id: str, name: str) -> Path:
    dest = UPLOADS / f"{uuid.uuid4().hex[:8]}_{name}"
    await bot.download(file_id, destination=dest)
    return dest


def _k(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


async def _process(message: types.Message, query: str) -> None:
    """Общий пайплайн: прогон графа со стримингом прогресса + расход токенов."""
    uid = str(message.from_user.id)
    _current_msg.set(message)  # канал для human-in-the-loop подтверждений
    status = await message.answer("🤔 Думаю…")
    tracker = TokenTracker()
    last_edit = {"ts": 0.0, "text": ""}

    async def _on_label(label: str) -> None:
        # промежуточные результаты: что агент делает сейчас + сколько токенов сожжено
        text = f"{label}\n🧮 {_k(tracker.total)} tok · ~${tracker.cost():.4f}"
        if text != last_edit["text"] and time.monotonic() - last_edit["ts"] > 1.5:
            last_edit.update(ts=time.monotonic(), text=text)
            try:
                await status.edit_text(text)
            except Exception:  # noqa: BLE001 — телеграм мог не принять edit, не критично
                pass

    try:
        cfg = {"configurable": {"thread_id": _thread(uid)}, "recursion_limit": 50,
               "callbacks": [tracker]}
        # aiogram обрабатывает апдейты конкурентно (каждый — своя asyncio-задача); без изоляции
        # бюджета два юзера сели бы на общий _default и reset() одного стёр бы счётчик другого (2a,
        # тот же баг, что закрыли в server.py — это второй вход).
        with run_context.request_scope(f"tg-{uid}-{uuid.uuid4().hex}", uid):
            result = await stream_with_progress(
                agent_app, {"query": query, "user_id": uid, "chat_history": []},
                config=cfg, on_label=_on_label,
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
        head += f"\n🧮 {_k(tracker.input)} in + {_k(tracker.output)} out · ~${tracker.cost():.4f}"
        add_alltime(tracker.input, tracker.output, tracker.calls)

        await status.delete()
        await message.answer(head)
        await _send_long(message, answer)
    except Exception as e:
        logger.error(f"agent error: {e}")
        await status.edit_text(f"⚠ Ошибка: {e}")


@dp.message(F.voice | F.audio)
async def handle_voice(message: types.Message):
    """Голосовой ввод: скачиваем → расшифровка fast-моделью → обычный пайплайн."""
    media = message.voice or message.audio
    note = await message.answer("🎙 Расшифровываю…")
    try:
        path = await _download(media.file_id, "voice.ogg")
        text = await asyncio.to_thread(transcribe_audio, str(path))
        await note.edit_text(f"🎙 «{text}»")
    except Exception as e:
        await note.edit_text(f"⚠ Не смог расшифровать: {e}")
        return
    await _process(message, text)


@dp.message(F.photo)
async def handle_photo(message: types.Message):
    """Картинка: vision-описание fast-моделью → в контекст запроса."""
    caption = message.caption or "Проанализируй изображение."
    note = await message.answer("🖼 Смотрю на изображение…")
    try:
        path = await _download(message.photo[-1].file_id, "photo.jpg")
        ctx = await asyncio.to_thread(attachment_context, [str(path)], caption)
        await note.delete()
    except Exception as e:
        await note.edit_text(f"⚠ Не смог обработать изображение: {e}")
        return
    await _process(message, f"{caption}\n\n=== ВЛОЖЕНИЯ ===\n{ctx}")


@dp.message(F.document)
async def handle_document(message: types.Message):
    """Файл-вложение: картинки → vision, текст → инлайн, прочее → путь для навыков."""
    doc = message.document
    caption = message.caption or f"Обработай приложенный файл {doc.file_name}."
    note = await message.answer(f"📎 Скачиваю {doc.file_name}…")
    try:
        path = await _download(doc.file_id, doc.file_name or "file.bin")
        ctx = await asyncio.to_thread(attachment_context, [str(path)], caption)
        await note.delete()
    except Exception as e:
        await note.edit_text(f"⚠ Не смог обработать файл: {e}")
        return
    await _process(message, f"{caption}\n\n=== ВЛОЖЕНИЯ ===\n{ctx}")


@dp.message(F.text)
async def handle_message(message: types.Message):
    # Если ждём ответ на открытый вопрос-уточнение — это он, а не новый запрос.
    uid = str(message.from_user.id)
    fut = _pending_text.get(uid)
    if fut is not None and not fut.done():
        fut.set_result(message.text.strip())
        return
    await _process(message, message.text)


async def main():
    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
