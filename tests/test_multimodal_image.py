"""Картинки отдаются модели НАПРЯМУЮ (image_url), а не через vision→текст-описание."""
import base64
import os
import tempfile

import src.graph.agent as A

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/1eYAAAAAElFTkSuQmCC")


def _png_path():
    p = os.path.join(tempfile.gettempdir(), "sea_test_px.png")
    with open(p, "wb") as f:
        f.write(_PNG)
    return p


def test_img_human_builds_multimodal_content():
    m = A._img_human("дай текст с картинки", [_png_path()])
    assert isinstance(m.content, list)
    assert [c["type"] for c in m.content] == ["text", "image_url"]
    assert m.content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_img_human_no_images_is_plain_text():
    m = A._img_human("просто текст", [])
    assert m.content == "просто текст"
    m2 = A._img_human("текст", None)
    assert m2.content == "текст"


def test_human_msg_reads_state_image_paths():
    m = A._human_msg({"image_paths": [_png_path()]}, "из state")
    assert isinstance(m.content, list) and len(m.content) == 2
    assert A._human_msg({}, "нет картинок").content == "нет картинок"


def test_vision_supported_flag_override(monkeypatch):
    """vision_supported: явный флаг vision_direct перебивает запомненные модальности."""
    import src.llm.llm as L

    monkeypatch.setattr(L, "_cli_override", lambda k, d=None: True if k == "vision_direct" else d)
    assert L.vision_supported() is True
    monkeypatch.setattr(L, "_cli_override", lambda k, d=None: False if k == "vision_direct" else d)
    assert L.vision_supported() is False


def test_modalities_from_openrouter_and_cache(monkeypatch):
    """Модальности берутся из OpenRouter /models; vision_supported читает ЗАПОМНЕННЫЙ кэш модели."""
    import json
    import src.llm.llm as L

    payload = {"data": [
        {"id": "google/gemini-x", "architecture": {"input_modalities": ["text", "image", "audio"]}},
        {"id": "deepseek/text-only", "architecture": {"input_modalities": ["text"]}},
    ]}

    class _R:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(L.urllib.request, "urlopen", lambda *a, **k: _R())
    assert L.fetch_modalities("google/gemini-x") == ["text", "image", "audio"]
    assert L.fetch_modalities("deepseek/text-only") == ["text"]
    assert L.fetch_modalities("nope/unknown") is None
    # vision_supported читает MODEL-AWARE кэш {model_id: [mods]} (флага нет)
    mid = L.model_for("fast")
    monkeypatch.setattr(L, "_cli_override",
                        lambda k, d=None: {mid: ["text", "image"]} if k == "model_modalities" else None)
    assert L.vision_supported() is True and L.supports_modality("audio") is False
    monkeypatch.setattr(L, "_cli_override",
                        lambda k, d=None: {mid: ["text"]} if k == "model_modalities" else None)
    assert L.vision_supported() is False
