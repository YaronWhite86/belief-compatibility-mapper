"""Unit tests for the SQLite result cache and rate limiter."""

from __future__ import annotations

import time

import pytest

from cache import RateLimiter, ResultCache
from models import TensionResult


class TestResultCache:
    @pytest.fixture()
    def cache(self, tmp_path):
        c = ResultCache(tmp_path / "test.db")
        yield c
        c.close()

    # -- embeddings ---------------------------------------------------

    def test_embedding_roundtrip(self, cache):
        cache.put_embedding("hello", "model-a", [0.1, 0.2])
        assert cache.get_embedding("hello", "model-a") == [0.1, 0.2]

    def test_embedding_miss(self, cache):
        assert cache.get_embedding("nonexistent", "model-a") is None

    def test_embedding_model_isolation(self, cache):
        cache.put_embedding("hello", "model-a", [1.0])
        assert cache.get_embedding("hello", "model-b") is None

    def test_embedding_overwrite(self, cache):
        cache.put_embedding("hello", "m", [1.0])
        cache.put_embedding("hello", "m", [2.0])
        assert cache.get_embedding("hello", "m") == [2.0]

    # -- tension results ----------------------------------------------

    def test_tension_roundtrip(self, cache):
        r = TensionResult(
            score=0.6, category="compatible_harmonious", justification="They agree."
        )
        cache.put_tension("A", "B", "sonnet", r)
        got = cache.get_tension("A", "B", "sonnet")
        assert got is not None
        assert got.score == 0.6
        assert got.category.value == "compatible_harmonious"

    def test_tension_order_independent(self, cache):
        r = TensionResult(
            score=-0.8, category="contradictory", justification="They clash."
        )
        cache.put_tension("X", "Y", "sonnet", r)
        assert cache.get_tension("Y", "X", "sonnet") is not None
        assert cache.get_tension("Y", "X", "sonnet").score == -0.8

    def test_tension_miss(self, cache):
        assert cache.get_tension("A", "B", "sonnet") is None

    def test_tension_model_isolation(self, cache):
        r = TensionResult(score=0.0, category="neutral", justification="Unrelated.")
        cache.put_tension("A", "B", "model-1", r)
        assert cache.get_tension("A", "B", "model-2") is None


class TestRateLimiter:
    def test_allows_within_budget(self):
        rl = RateLimiter(max_rpm=10)
        start = time.monotonic()
        for _ in range(10):
            rl.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # should be near-instant

    def test_sleeps_when_exhausted(self):
        rl = RateLimiter(max_rpm=2)
        # Inject two timestamps that are almost 60s old so the
        # sleep is very short (~0.1s) rather than a full 60s.
        now = time.monotonic()
        rl._timestamps.append(now - 59.95)
        rl._timestamps.append(now - 59.90)

        start = time.monotonic()
        rl.wait()
        elapsed = time.monotonic() - start

        # Should have slept briefly (until the oldest stamp expires)
        assert elapsed < 1.0
        # After sleeping, old timestamps are evicted and one new one
        # is added. Depending on timing, the second injected timestamp
        # (now - 59.90) may or may not have expired yet, so we allow
        # 1 or 2 timestamps remaining.
        assert 1 <= len(rl._timestamps) <= 2

    def test_deque_eviction(self):
        rl = RateLimiter(max_rpm=100)
        # Inject an old timestamp
        rl._timestamps.append(time.monotonic() - 120)
        rl.wait()
        # The old timestamp should have been evicted
        assert len(rl._timestamps) == 1
