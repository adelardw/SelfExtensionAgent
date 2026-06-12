"""TurboVec-обёртка VectorIndex: корректность NN, allowlist, нормировка. Пропуск без turbovec."""
import numpy as np
import pytest

from src.memory.vector_index import VectorIndex, turbovec_available, MIN_VECTORS

pytestmark = pytest.mark.skipif(not turbovec_available(), reason="turbovec не установлен")


def _build(scale_query=1.0, scale_noise=0.05, n=200, dim=128, dup_id=7, seed=3):
    rng = np.random.default_rng(seed)
    base = (rng.standard_normal(dim)).astype("float32")
    vi = VectorIndex(dim=dim, bit_width=4)
    for i in range(n):
        v = (base + scale_noise * rng.standard_normal(dim)).astype("float32") if i == dup_id \
            else (rng.standard_normal(dim)).astype("float32")
        vi.add(i, v.tolist())
    return vi, base, dup_id


def test_nearest_neighbor_correct():
    vi, base, dup = _build()
    res = vi.search(base.tolist(), k=3)
    assert res and res[0][0] == dup


def test_normalization_makes_ip_cosine():
    # вход ненормирован и в другом масштабе — нормировка в обёртке должна спасти ранг
    vi, base, dup = _build()
    res = vi.search((base * 0.13).tolist(), k=3)
    assert res and res[0][0] == dup


def test_allowlist_restricts_results():
    vi, base, dup = _build()
    res = vi.search(base.tolist(), k=2, allowed=[5, dup])
    ids = [i for i, _ in res]
    assert dup in ids and all(i in (5, dup) for i in ids)


def test_active_threshold():
    vi = VectorIndex(dim=16, bit_width=4)
    for i in range(MIN_VECTORS - 1):
        vi.add(i, np.random.default_rng(i).standard_normal(16).astype("float32").tolist())
    assert not vi.active
    vi.add(999, np.ones(16, dtype="float32").tolist())
    assert vi.active
