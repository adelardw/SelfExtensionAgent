"""
Анти-галлюцинация БЕЗ регэкспов/ключевых слов.

Политика проекта: маршрутизацию и семантику решает модель/эмбеддинги, не лексиконные регэкспы
(см. intent.py, semantics.py). Прежние русско-центричные регэкспы детекторов убраны. Теперь:

  • PAYWALL (стена подписки/входа на странице) — embedding-классификатор: cosine-kNN к
    мультиязычным seed'ам, КОНТРАСТИВНО (позитивы vs негативы), любой язык страницы. Бежит в
    act-ветке воспроизведения (раз на play-попытку, не в петле) → один дешёвый эмбеддинг.
    Контраст (порог + маржа над лучшим НЕГАТИВОМ) защищает близкие легитимные страницы
    («сейчас играет» / «смотреть бесплатно» ≠ «оформите подписку»).

  • Ложный отказ «нет доступа» и мета-заглушка «я перечислил/список выше» — НЕ здесь: их
    семантически флагает финальный LLM-валидатор (validation_node + ValidationResult), без
    отдельного вызова (он и так судит финал, на любом языке). «Ловлю галлюцинации» делает
    модель, а не список слов.

  • DEGENERATE (вырожденный повтор, «I'm Sorry» ×58) — структурный счётчик (доля уникальных
    слов), языко-независим по конструкции, не регэксп/лексикон. Дешёвый предохранитель.

Единый источник для рабочего графа (agent.py) и экспериментального (agent_experimental.py).
"""
from __future__ import annotations

import re
from typing import Optional

from src.memory.embedder import cosine


def _cfg_signals() -> dict:
    d = {"paywall_threshold": 0.58, "paywall_margin": 0.05,
         "error_threshold": 0.55, "error_margin": 0.05,
         "media_threshold": 0.55, "media_margin": 0.05}
    try:
        from omegaconf import OmegaConf
        c = OmegaConf.load("config.yml").get("semantic_signals", {}) or {}
        for k in d:
            d[k] = float(c.get(k, d[k]))
    except Exception:  # noqa: BLE001
        pass
    return d


_SIG_CFG = _cfg_signals()

# Мультиязычные seed'ы paywall: POS — текст-носитель стены подписки/входа; NEG — близкие, но
# ЛЕГИТИМНЫЕ страницы (контраст, чтобы «сейчас играет»/«бесплатно»/«меню» не считать стеной).
_PAYWALL_POS = [
    "subscription required to watch this", "оформите подписку, чтобы смотреть",
    "members only, sign in to continue watching", "войдите в аккаунт, чтобы смотреть",
    "available to rent or buy", "upgrade to premium to watch",
    "需要订阅才能观看", "登录后才能观看", "会員限定のコンテンツです",
    "اشترك للمشاهدة", "멤버십 전용 콘텐츠입니다", "para ver necesitas suscripción",
    "nur für abonnenten verfügbar", "réservé aux abonnés",
]
_PAYWALL_NEG = [
    "now playing", "сейчас играет", "видео воспроизводится", "free to watch", "смотреть бесплатно",
    "search results", "результаты поиска", "home page navigation menu", "главное меню сайта",
    "add to playlist", "следующее видео", "комментарии к видео",
]


class _ContrastiveSignal:
    """Контрастивный cosine-kNN детектор по мультиязычным seed'ам (POS vs NEG). Любой язык,
    без регэкспов/лексикона. Эмбеддер инъектируем (офлайн-тест механики без сети). Seed'ы
    эмбеддятся лениво и кэшируются на процесс. Один класс — все сигналы (paywall/error/media)."""

    def __init__(self, pos_seeds: list, neg_seeds: list, embedder=None):
        self._pos_seeds = pos_seeds
        self._neg_seeds = neg_seeds
        self._embedder = embedder
        self._pos_v: Optional[list] = None
        self._neg_v: Optional[list] = None

    def _emb(self):
        if self._embedder is None:
            from pathlib import Path
            from omegaconf import OmegaConf
            from src.memory.embedder import build_embedder
            c = OmegaConf.load("config.yml") if Path("config.yml").exists() else {}
            self._embedder = build_embedder(
                (c.get("memory", {}) or {}).get("embeddings", False) if c else False,
                (c.get("memory", {}) or {}).get("embedding_model") if c else None,
            )
        return self._embedder

    @property
    def enabled(self) -> bool:
        try:
            return bool(self._emb().enabled)
        except Exception:  # noqa: BLE001
            return False

    def _ensure_seeds(self) -> None:
        if self._pos_v is not None:
            return
        emb = self._emb()
        self._pos_v = [v for v in (emb.embed(t) for t in self._pos_seeds) if v]
        self._neg_v = [v for v in (emb.embed(t) for t in self._neg_seeds) if v]

    def fires(self, text: str, threshold: float, margin: float) -> bool:
        if not (text or "").strip() or not self.enabled:
            return False
        try:
            self._ensure_seeds()
            if not self._pos_v:
                return False
            qv = self._emb().embed((text or "")[:1500])
            if not qv:
                return False
            pos = max((cosine(qv, v) for v in self._pos_v), default=0.0)
            neg = max((cosine(qv, v) for v in (self._neg_v or [])), default=0.0)
            return pos >= threshold and (pos - neg) >= margin
        except Exception:  # noqa: BLE001
            return False


class _PaywallEmbed(_ContrastiveSignal):
    """Back-compat обёртка: paywall с захардкоженными мультиязычными сидами."""

    def __init__(self, embedder=None):
        super().__init__(_PAYWALL_POS, _PAYWALL_NEG, embedder)


# ── СТРАНИЦА-ОШИБКА (404/не найдена) — внешний контент чужой страницы, любой язык ──
_ERROR_POS = [
    "404 not found", "page not found", "страница не найдена", "этой страницы не существует",
    "ничего не нашлось", "the requested url was not found", "页面未找到", "ページが見つかりません",
    "página no encontrada", "seite nicht gefunden", "오류 페이지를 찾을 수 없습니다", "الصفحة غير موجودة",
]
_ERROR_NEG = [
    "now playing", "сейчас играет", "search results", "результаты поиска",
    "video is playing", "статья", "article content", "главная страница сайта", "товары в наличии",
]

# ── ВОСПРОИЗВЕДЕНИЕ РЕАЛЬНО ИДЁТ (фолбэк для extension-прозы; in-repo путь даёт структурный флаг) ──
_MEDIA_POS = [
    "audio is now playing", "звук играет", "video is playing", "сейчас воспроизводится",
    "playback started", "now playing", "видео воспроизводится", "再生中", "正在播放", "reproduciendo ahora",
]
_MEDIA_NEG = [
    "paused", "на паузе", "press play to start", "нажмите воспроизведение", "play button available",
    "track listing", "список треков", "добавить в плейлист", "video is paused", "остановлено",
]

# Синглтоны на процесс (кэш seed-эмбеддингов). cosine — из того же эмбеддера, что память/intent.
_PAYWALL: Optional[_ContrastiveSignal] = None
_ERROR: Optional[_ContrastiveSignal] = None
_MEDIA: Optional[_ContrastiveSignal] = None


def _paywall_detector() -> _ContrastiveSignal:
    global _PAYWALL
    if _PAYWALL is None:
        _PAYWALL = _ContrastiveSignal(_PAYWALL_POS, _PAYWALL_NEG)
    return _PAYWALL


def _error_detector() -> _ContrastiveSignal:
    global _ERROR
    if _ERROR is None:
        _ERROR = _ContrastiveSignal(_ERROR_POS, _ERROR_NEG)
    return _ERROR


def _media_detector() -> _ContrastiveSignal:
    global _MEDIA
    if _MEDIA is None:
        _MEDIA = _ContrastiveSignal(_MEDIA_POS, _MEDIA_NEG)
    return _MEDIA


def is_error_page(text: str) -> bool:
    """Страница-ошибка (404/«не найдена») — embedding-контраст, любой язык. На play-намерении
    нельзя заявлять воспроизведение, если под плеером страница-ошибка. Без эмбеддера → False."""
    return _error_detector().fires(text or "", _SIG_CFG["error_threshold"], _SIG_CFG["error_margin"])


def is_media_playing(text: str) -> bool:
    """ФОЛБЭК-детект «звук реально идёт» для extension-прозы (in-repo browser_session даёт точный
    структурный флаг MEDIA_PLAYING, его проверяют первым). Контраст POS «играет» vs NEG «пауза/
    список треков» — не путать с листингом. Без эмбеддера → False (act даёт честный статус)."""
    return _media_detector().fires(text or "", _SIG_CFG["media_threshold"], _SIG_CFG["media_margin"])


def is_paywall(text: str) -> bool:
    """Стена подписки/входа на странице (контент платный/закрыт) — embedding-классификатор,
    любой язык. Без эмбеддера → False (не блокируем play по догадке; act всё равно даёт честный
    статус, если плей не пошёл)."""
    return _paywall_detector().fires(text or "", _SIG_CFG["paywall_threshold"], _SIG_CFG["paywall_margin"])


def is_degenerate(text: str) -> bool:
    """Вырожденный повтор в ответе (модель залипла: «I'm Sorry» ×58) — галлюцинация, не данные.
    Признак: большой объём при крошечной доле уникальных слов. Структурный (не лексикон/регэксп),
    языко-независимый по конструкции, без LLM."""
    # Чистим нумерацию списка и пунктуацию: «57.», «(I'm», «Sorry)» не должны маскировать
    # повтор уникальными номерами строк (иначе 1..58 раздували долю уникальных слов).
    words = [w for w in re.sub(r"[\d().,\[\]]+", " ", text or "").split() if w]
    if len(words) < 60:
        return False
    uniq = len({w.lower() for w in words})
    return uniq / len(words) < 0.15
