"""
Универсальный (любой язык) embedding-kNN роутер интентов — «кодбук правильных роутов».

Заменяет русско-лексиконные регэкспы маршрутизации мультиязычным классификатором:
мультиязычность — от эмбеддингов («play music» и «включи музыку» → близкие векторы).
РАСТЁТ ИЗ ФИДБЕК-ЛУПА: валидированный успешный прогон добавляет (запрос → сработавший
маршрут) экземпляром (reflect → add_exemplar) — кодбук правильных роутов накапливается.

Лейблы маршрута (сигналы для гейтов фронта):
- web_grounding    — нужны СВЕЖИЕ внешние факты (где купить/цена/адрес/лучшие/новости/как оформить)
- physical_browser — действие в браузере юзера (открыть сайт/войти/корзина/клик)
- play_media       — воспроизвести музыку/видео/фильм
- self_contained   — ответ из знаний/рассуждения (мат/объяснение/код/приветствие)

Cold-start: курируемые мультиязычные примеры (RU+EN). Эмбеддинги seed'а кэшируются в
data/intent_codebook.json (вычисляются 1 раз). Деградация: эмбеддингов нет / кодбук пуст /
низкая уверенность → классификатор возвращает None → caller берёт регэксп-fallback.

Hot-path: классификация переиспользует УЖЕ посчитанный в recall эмбеддинг запроса
(state['query_emb']) → ноль лишних сетевых вызовов.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .memory.embedder import build_embedder, cosine

CODEBOOK_FILE = Path(os.getenv("AGENT_INTENT_CODEBOOK") or "data/intent_codebook.json")
MIN_SIM = 0.45          # ниже — «не уверен» → fallback на регэксп
MAX_PER_LABEL = 60      # потолок выученных экземпляров на лейбл (анти-переполнение, LRU по ts)

LABELS = ("web_grounding", "physical_browser", "play_media", "self_contained")

# Курируемый мультиязычный seed (RU+EN). Иллюстрирует КЛАССЫ маршрута, не конкретные сценарии.
_SEED: dict[str, list[str]] = {
    "web_grounding": [
        "где купить недорогие наушники", "where can I buy cheap headphones",
        "лучшие рестораны суши в москве", "best sushi places near me",
        "сколько сейчас стоит биткоин", "current price of bitcoin",
        "как оформить загранпаспорт", "how to apply for a passport",
        "свежие новости про ИИ", "latest news about AI",
        "адрес ближайшей аптеки", "address of the nearest pharmacy",
    ],
    "physical_browser": [
        "открой сайт github и залогинься", "open my email and log in",
        "добавь товар в корзину на озоне", "add this item to my cart",
        "зайди в мой личный кабинет", "go to my account page",
        "нажми кнопку оформить заказ", "click the checkout button on the site",
    ],
    "play_media": [
        "включи музыку группы radiohead", "play some jazz music",
        "поставь фильм дюна", "play the movie Dune", "включи трек на ютубе",
        "play this song on spotify", "запусти видео с котиками",
    ],
    "self_contained": [
        "посчитай среднюю скорость поезда", "what is 17 times 23",
        "объясни что такое рекурсия", "explain how recursion works",
        "напиши функцию факториала на python", "write a python factorial function",
        "привет как дела", "hello how are you", "переведи фразу на английский",
    ],
}


class IntentRouter:
    """Кодбук маршрутов + cosine-kNN классификация. Лениво эмбеддит seed и кэширует."""

    def __init__(self) -> None:
        self._embedder = None
        self._entries: list[dict] = []   # [{text,label,emb,ts,learned}]
        self._loaded = False

    # ── lazy init ────────────────────────────────────────────────────
    def _emb(self):
        if self._embedder is None:
            from omegaconf import OmegaConf
            c = OmegaConf.load("config.yml") if Path("config.yml").exists() else {}
            self._embedder = build_embedder(
                (c.get("memory", {}) or {}).get("embeddings", False) if c else False,
                (c.get("memory", {}) or {}).get("embedding_model") if c else None,
            )
        return self._embedder

    @property
    def enabled(self) -> bool:
        return self._emb().enabled

    def _load(self) -> None:
        if self._loaded:
            return
        if CODEBOOK_FILE.exists():
            try:
                self._entries = json.loads(CODEBOOK_FILE.read_text(encoding="utf-8")).get("entries", [])
            except Exception:  # noqa: BLE001
                self._entries = []
        # seed (один раз) — только если эмбеддер жив
        if not any(not e.get("learned") for e in self._entries) and self.enabled:
            seed = [(t, lbl) for lbl, ts in _SEED.items() for t in ts]
            vecs = self._embed_batch([t for t, _ in seed])
            if vecs:
                for (t, lbl), v in zip(seed, vecs):
                    if v:
                        self._entries.append({"text": t, "label": lbl, "emb": v, "ts": time.time(), "learned": False})
                self._save()
        self._loaded = True

    def _embed_batch(self, texts: list[str]) -> Optional[list]:
        emb = self._emb()
        if not emb.enabled:
            return None
        out = []
        for t in texts:  # embedder.embed по одному (батч-API не у всех провайдеров одинаков)
            out.append(emb.embed(t))
        return out

    def _save(self) -> None:
        try:
            CODEBOOK_FILE.parent.mkdir(parents=True, exist_ok=True)
            CODEBOOK_FILE.write_text(json.dumps({"entries": self._entries}, ensure_ascii=False), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # ── classify ─────────────────────────────────────────────────────
    def classify(self, text: str, qvec: Optional[list] = None) -> Optional[dict]:
        """
        Возвращает {'label','score','scores':{label:max_sim}} или None (нельзя классифицировать:
        нет эмбеддингов / пустой кодбук / низкая уверенность → caller берёт регэксп-fallback).
        qvec — предвычисленный эмбеддинг запроса (из recall) — переиспользуем, не эмбеддим заново.
        """
        if not self.enabled:
            return None
        self._load()
        if not self._entries:
            return None
        if qvec is None:
            qvec = self._emb().embed(text)
        if not qvec:
            return None
        per_label: dict[str, float] = {l: 0.0 for l in LABELS}
        for e in self._entries:
            emb = e.get("emb")
            if not emb:
                continue
            s = cosine(qvec, emb)
            lbl = e.get("label", "")
            if lbl in per_label and s > per_label[lbl]:
                per_label[lbl] = s   # 1-NN на лейбл (макс. близость к классу) — робастно на малом кодбуке
        best = max(per_label, key=per_label.get)
        if per_label[best] < MIN_SIM:
            return None  # не уверен → fallback
        return {"label": best, "score": per_label[best], "scores": per_label}

    # ── grow from feedback loop ──────────────────────────────────────
    def add_exemplar(self, text: str, label: str, qvec: Optional[list] = None) -> None:
        """Кодбук растёт из фидбек-лупа: валидированный прогон → (запрос→сработавший маршрут)."""
        if label not in LABELS or not (text or "").strip():
            return
        self._load()
        if not self.enabled:
            return
        if qvec is None:
            qvec = self._emb().embed(text)
        if not qvec:
            return
        key = text.strip()[:80].lower()
        # дедуп по началу запроса в этом лейбле
        self._entries = [e for e in self._entries
                         if not (e.get("learned") and e.get("label") == label
                                 and e.get("text", "").strip()[:80].lower() == key)]
        self._entries.append({"text": text[:200], "label": label, "emb": qvec, "ts": time.time(), "learned": True})
        # LRU-cap выученных на лейбл (seed не трогаем)
        learned = [e for e in self._entries if e.get("learned") and e.get("label") == label]
        if len(learned) > MAX_PER_LABEL:
            learned.sort(key=lambda e: e.get("ts", 0))
            drop = {id(e) for e in learned[: len(learned) - MAX_PER_LABEL]}
            self._entries = [e for e in self._entries if id(e) not in drop]
        self._save()


# Синглтон роутера (кэш кодбука/эмбеддера на процесс).
_ROUTER: Optional[IntentRouter] = None


def get_router() -> IntentRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = IntentRouter()
    return _ROUTER


# ── Корпус маршрутов для БУДУЩЕГО contrastive-обучения локального эмбеддера ──────────
# Хранит (текст, маршрут, reward 0/1) — позитивы И негативы. Append-only, ОТДЕЛЁН от
# live-кодбука (тот растёт только на успехах и влияет на retrieval; корпус — сырьё для
# обучения и НЕ влияет на рантайм). Метка = (маршрут + исход), НЕ «маршрут верный/неверный»
# (провал мог быть из-за исполнения) — это шум, который чистится при обучении.
_CORPUS_DB = os.getenv("AGENT_ROUTE_CORPUS") or "data/route_examples.db"
_corpus_conn: Optional[sqlite3.Connection] = None


def _corpus() -> sqlite3.Connection:
    global _corpus_conn
    if _corpus_conn is None:
        Path(_CORPUS_DB).parent.mkdir(parents=True, exist_ok=True)
        _corpus_conn = sqlite3.connect(_CORPUS_DB, check_same_thread=False)
        _corpus_conn.execute(
            "CREATE TABLE IF NOT EXISTS route_examples ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts REAL, "
            "text TEXT, route TEXT, reward INTEGER)")
        _corpus_conn.commit()
    return _corpus_conn


def log_route_example(text: str, route: str, reward: int, user_id: str = "") -> None:
    """Append (текст, маршрут, reward 0/1) в корпус. Сырьё для будущего contrastive-обучения
    локального эмбеддера; на live-роутинг НЕ влияет. Тихо игнорирует сбои (не ломает reflect)."""
    if not (text or "").strip() or route not in LABELS:
        return
    try:
        c = _corpus()
        c.execute("INSERT INTO route_examples (user_id, ts, text, route, reward) VALUES (?,?,?,?,?)",
                  (user_id or "default", time.time(), text[:500], route, 1 if reward else 0))
        c.commit()
    except Exception:  # noqa: BLE001
        pass


def corpus_stats() -> dict:
    """Сводка корпуса (для диагностики/решения «пора обучать»): всего / pos / neg / по маршрутам."""
    try:
        c = _corpus()
        rows = c.execute("SELECT route, reward, COUNT(*) FROM route_examples GROUP BY route, reward").fetchall()
    except Exception:  # noqa: BLE001
        return {"total": 0}
    out: dict = {"total": 0, "pos": 0, "neg": 0, "by_route": {}}
    for route, reward, cnt in rows:
        out["total"] += cnt
        out["pos" if reward else "neg"] += cnt
        br = out["by_route"].setdefault(route, {"pos": 0, "neg": 0})
        br["pos" if reward else "neg"] += cnt
    return out
