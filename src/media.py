"""
Работа с медиа-вложениями: картинки (vision) и голос (транскрипция).

Без отдельных платных STT/vision-сервисов: gemini-2.5-flash-lite мультимодален
(изображения + аудио на входе) и уже является fast-тиром агента — то есть
вложения стоят те же копейки, что и обычный запрос.

  describe_image(path, question)  → текстовое описание/анализ изображения
  transcribe_audio(path)          → расшифровка голосового сообщения
  attachment_context(paths, q)    → готовый блок контекста по списку вложений
"""
from __future__ import annotations

import base64
import mimetypes
import shutil
import subprocess
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from .llm import chat

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic"}
AUDIO_EXTS = {".ogg", ".oga", ".mp3", ".wav", ".m4a", ".aiff", ".flac"}
TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".yml", ".yaml", ".py", ".log", ".html", ".xml"}

MAX_INLINE_TEXT = 6000  # сколько символов текстового файла инлайнить в контекст


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def describe_image(path: str | Path, question: str = "") -> str:
    """Vision-анализ изображения fast-моделью. question — на что обратить внимание."""
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    prompt = (
        "Опиши это изображение подробно и структурно: что на нём, весь видимый текст "
        "(перепиши точно), числа, таблицы, UI-элементы. Это описание будет контекстом "
        "для агента, отвечающего на запрос пользователя."
    )
    if question:
        prompt += f"\nЗапрос пользователя (учти при описании): {question}"
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(p)}"}},
    ])
    resp = chat("fast").invoke([msg])
    return resp.content if hasattr(resp, "content") else str(resp)


def _to_mp3(path: Path) -> Path:
    """Конвертация в mp3 через ffmpeg (telegram-голосовые приходят в ogg/opus)."""
    if not shutil.which("ffmpeg"):
        return path  # попробуем отдать как есть
    out = Path(tempfile.mkstemp(suffix=".mp3")[1])
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-b:a", "48k", str(out)],
        capture_output=True, timeout=60,
    )
    return out if res.returncode == 0 and out.stat().st_size > 0 else path


def transcribe_audio(path: str | Path) -> str:
    """Расшифровка аудио fast-моделью (audio-вход gemini). Возвращает чистый текст."""
    p = Path(path)
    fmt = p.suffix.lstrip(".").lower()
    if fmt not in ("mp3", "wav"):
        p = _to_mp3(p)
        fmt = p.suffix.lstrip(".").lower() or "mp3"
    msg = HumanMessage(content=[
        {"type": "text", "text": "Расшифруй это голосовое сообщение. Верни ТОЛЬКО текст сказанного, без комментариев."},
        {"type": "input_audio", "input_audio": {"data": _b64(p), "format": fmt}},
    ])
    resp = chat("fast").invoke([msg])
    return (resp.content if hasattr(resp, "content") else str(resp)).strip()


def attachment_context(paths: list[str | Path], question: str = "") -> str:
    """
    Готовит блок контекста по вложениям: картинки — vision-описание, текстовые файлы —
    инлайн содержимого (с обрезкой), прочее — путь+размер (deliberate-путь откроет
    навыками file_operations/text_file_processor).
    """
    blocks = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            blocks.append(f"[Вложение {p.name}: файл не найден]")
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            try:
                desc = describe_image(p, question)
                blocks.append(f"[Изображение {p.name} (путь: {p})]\nОписание:\n{desc}")
            except Exception as e:  # noqa: BLE001
                blocks.append(f"[Изображение {p.name} (путь: {p}) — vision не сработал: {e}]")
        elif ext in TEXT_EXTS:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                cut = text[:MAX_INLINE_TEXT]
                tail = f"\n…(обрезано, всего {len(text)} симв.; полный файл: {p})" if len(text) > MAX_INLINE_TEXT else ""
                blocks.append(f"[Файл {p.name} (путь: {p})]\n{cut}{tail}")
            except Exception as e:  # noqa: BLE001
                blocks.append(f"[Файл {p.name} (путь: {p}) — не читается: {e}]")
        else:
            blocks.append(
                f"[Файл {p.name} (путь: {p}, {p.stat().st_size} байт) — бинарный/другой формат; "
                f"открой инструментами при необходимости]"
            )
    return "\n\n".join(blocks)
