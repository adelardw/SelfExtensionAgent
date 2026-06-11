"""
Канонический лёгкий retrieval для core-модулей (ОДИН механизм над разными корпусами:
навыки, память, и т.п. — как в CLAUDE.md). BM25S при наличии, иначе token-overlap.

web_search держит СВОЮ инлайн-копию намеренно: навык грузится через exec_module в
песочнице, где `import src.*` может быть недоступен на этапе loadability-гейта. Здесь —
для надёжных core-модулей (импорт src гарантирован).
"""
from __future__ import annotations

import re

try:
    import bm25s
    _BM25 = True
except Exception:  # noqa: BLE001
    _BM25 = False


def _token_overlap_rank(docs: list[str], query: str, top: int) -> list[int]:
    """Фолбэк-ранкер без BM25S: пересечение токенов запроса и документа."""
    q = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    if not q:
        return list(range(min(top, len(docs))))
    scored = []
    for i, d in enumerate(docs):
        dt = set(re.findall(r"\w+", d.lower()))
        ov = len(q & dt)
        if ov:
            scored.append((ov + ov / (1 + len(dt) ** 0.5), i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:top]]


def bm25_rank(docs: list[str], query: str, top: int) -> list[int]:
    """
    Индексы топ-`top` РЕЛЕВАНТНЫХ документов под запрос (score>0), в порядке ранга.
    Языко-агностичная токенизация (stopwords/stemmer off) — корректно для рус/смешанного.
    Нерелевантные (score 0) НЕ возвращаем. Фолбэк — token-overlap.
    """
    if not docs:
        return []
    if not _BM25:
        return _token_overlap_rank(docs, query, top)
    try:
        tok = dict(stopwords=None, stemmer=None, show_progress=False)
        retr = bm25s.BM25()
        retr.index(bm25s.tokenize(docs, **tok), show_progress=False)
        results, scores = retr.retrieve(bm25s.tokenize(query, **tok),
                                        k=min(top, len(docs)), show_progress=False)
        idx = [int(i) for i, sc in zip(results[0], scores[0]) if sc > 0]
        return idx or _token_overlap_rank(docs, query, top)
    except Exception:  # noqa: BLE001
        return _token_overlap_rank(docs, query, top)
