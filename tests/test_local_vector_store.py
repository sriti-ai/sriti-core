from __future__ import annotations

import numpy as np
import pytest

from sriti.core.local_vector_store import LocalVectorStore
from tests.fake_redis import FakeRedis

pytestmark = pytest.mark.asyncio


def _vec(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


async def test_add_then_search_finds_closest_by_cosine_similarity():
    store = LocalVectorStore(FakeRedis(), key_prefix="test:")
    await store.add("test:a", _vec(1.0, 0.0), {"label": "a"})
    await store.add("test:b", _vec(0.0, 1.0), {"label": "b"})

    results = store.search(_vec(0.9, 0.1), k=1)
    assert len(results) == 1
    similarity, meta = results[0]
    assert meta["label"] == "a"
    assert similarity > 0.9


async def test_search_respects_filters():
    store = LocalVectorStore(FakeRedis(), key_prefix="test:")
    await store.add("test:a", _vec(1.0, 0.0), {"tenant_id": "biz1", "label": "a"})
    await store.add("test:b", _vec(1.0, 0.0), {"tenant_id": "biz2", "label": "b"})

    results = store.search(_vec(1.0, 0.0), k=5, filters={"tenant_id": "biz1"})
    assert len(results) == 1
    assert results[0][1]["label"] == "a"


async def test_load_rebuilds_index_from_redis():
    redis_client = FakeRedis()
    store1 = LocalVectorStore(redis_client, key_prefix="test:")
    await store1.add("test:a", _vec(1.0, 0.0), {"label": "a"})

    store2 = LocalVectorStore(redis_client, key_prefix="test:")
    assert len(store2) == 0
    await store2.load()
    assert len(store2) == 1

    results = store2.search(_vec(1.0, 0.0), k=1)
    assert results[0][1]["label"] == "a"


async def test_expired_records_excluded_from_search():
    store = LocalVectorStore(FakeRedis(), key_prefix="test:")
    await store.add("test:a", _vec(1.0, 0.0), {"label": "a"}, ttl_seconds=-1)

    results = store.search(_vec(1.0, 0.0), k=5)
    assert results == []


async def test_load_is_fail_open_on_error():
    class BrokenRedis(FakeRedis):
        async def scan_iter(self, match: str = "*"):
            raise ConnectionError("boom")
            yield  # pragma: no cover — makes this an async generator

    store = LocalVectorStore(BrokenRedis(), key_prefix="test:")
    await store.load()  # must not raise
    assert len(store) == 0
