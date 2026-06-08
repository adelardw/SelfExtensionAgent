"""
VectorIndex — обёртка над TurboVec (IdMapIndex) для семантического recall.

Graceful by design: если turbovec не установлен ИЛИ эмбеддинги выключены,
объект не создаётся, и MemoryStore падает на линейный скоринг по токенам.

Зачем IdMapIndex: внешние id = rowid из SQLite, поэтому per-user фильтрация
делается через allowlist (передаём id эпизодов пользователя) без отдельных
индексов на каждого. Квантизация 2–4 бита + SIMD — поиск дешёвый даже на CPU.

На малых объёмах ANN-квантизация бессмысленна и может быть неустойчива, поэтому
индекс реально используется только начиная с MIN_VECTORS; иначе — линейный путь.
"""
from __future__ import annotations

from typing import Optional

# Порог, с которого имеет смысл идти через ANN, а не линейно.
MIN_VECTORS = 32

try:
    import numpy as np
    from turbovec import IdMapIndex

    _AVAILABLE = True
except Exception:  # noqa: BLE001
    _AVAILABLE = False


def turbovec_available() -> bool:
    return _AVAILABLE


class VectorIndex:
    """Тонкая обёртка: add(id, vec) / search(qvec, k, allowed_ids)."""

    def __init__(self, dim: int, bit_width: int = 4):
        if not _AVAILABLE:
            raise RuntimeError("turbovec не установлен")
        self.dim = dim
        self._index = IdMapIndex(dim=dim, bit_width=bit_width)
        self._ids: set[int] = set()

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def active(self) -> bool:
        """Достаточно ли векторов, чтобы доверять ANN-поиску."""
        return len(self._ids) >= MIN_VECTORS

    def add(self, row_id: int, vec: list[float]) -> None:
        if not vec or len(vec) != self.dim or row_id in self._ids:
            return
        try:
            v = np.asarray([vec], dtype=np.float32)
            ids = np.asarray([row_id], dtype=np.uint64)
            self._index.add_with_ids(v, ids)
            self._ids.add(row_id)
        except Exception as e:  # noqa: BLE001
            print(f"[VectorIndex] add failed: {e}")

    def search(self, qvec: list[float], k: int, allowed: Optional[list[int]] = None) -> list[tuple[int, float]]:
        """Возвращает [(row_id, score)] по убыванию близости. [] при любой ошибке."""
        if not qvec or len(qvec) != self.dim or not self._ids:
            return []
        try:
            q = np.asarray([qvec], dtype=np.float32)
            kwargs = {}
            if allowed:
                allow = np.asarray([i for i in allowed if i in self._ids], dtype=np.uint64)
                if allow.size == 0:
                    return []
                kwargs["allowlist"] = allow
            scores, ids = self._index.search(q, k=min(k, len(self._ids)), **kwargs)
            return [(int(i), float(s)) for s, i in zip(scores[0], ids[0])]
        except Exception as e:  # noqa: BLE001
            print(f"[VectorIndex] search failed: {e}")
            return []
