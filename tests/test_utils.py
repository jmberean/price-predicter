"""Tests for utility modules (cache, device) - Phase 1 & 2."""

import pytest
import time
from datetime import datetime, timedelta

from aetheris_oracle.utils.cache import SimpleCache, make_cache_key, cached_fetch
from aetheris_oracle.utils.device import get_available_device


class TestSimpleCache:
    """Tests for SimpleCache."""

    def test_cache_set_get(self):
        """Test basic cache set and get."""
        cache = SimpleCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss_returns_none(self):
        """Test cache miss returns None."""
        cache = SimpleCache()
        assert cache.get("nonexistent") is None

    def test_cache_tracks_hits_and_misses(self):
        """Test cache statistics tracking."""
        cache = SimpleCache()

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_eviction_when_full(self):
        """Test LRU eviction when cache is full."""
        cache = SimpleCache(max_size=2, ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_expiration_after_ttl(self):
        """Test cache entries expire after TTL."""
        cache = SimpleCache(max_size=10, ttl_seconds=1)  # 1 second TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        assert cache.get("key1") is None  # Expired

    def test_cache_updates_lru_on_access(self):
        """Test that accessing a key updates LRU order."""
        cache = SimpleCache(max_size=2, ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access key1 (makes it most recently used)
        cache.get("key1")

        # Add key3 (should evict key2, not key1)
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"

    def test_cache_clear(self):
        """Test cache clear removes all entries."""
        cache = SimpleCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.stats()["size"] == 0

    def test_cache_stats_structure(self):
        """Test cache stats returns correct structure."""
        cache = SimpleCache(max_size=100, ttl_seconds=3600)

        stats = cache.stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "total_requests" in stats
        assert "hit_rate" in stats
        assert "size" in stats
        assert "max_size" in stats
        assert stats["max_size"] == 100


class TestCacheHelpers:
    """Tests for cache helper functions."""

    def test_make_cache_key(self):
        """Test cache key generation."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 7, 12, 0, 0)

        key = make_cache_key("BTC-USD", start, end)

        assert "BTC-USD" in key
        assert "2024-01-01" in key
        assert "2024-01-07" in key

    def test_cached_fetch_on_cache_hit(self):
        """Test cached_fetch returns cached value without calling fetch_fn."""
        cache = SimpleCache()
        cache.set("test_key", "cached_value")

        fetch_called = False

        def fetch_fn():
            nonlocal fetch_called
            fetch_called = True
            return "fresh_value"

        result = cached_fetch(cache, "test_key", fetch_fn)

        assert result == "cached_value"
        assert not fetch_called  # Fetch should not be called

    def test_cached_fetch_on_cache_miss(self):
        """Test cached_fetch calls fetch_fn and caches result."""
        cache = SimpleCache()

        def fetch_fn():
            return "fresh_value"

        result = cached_fetch(cache, "missing_key", fetch_fn)

        assert result == "fresh_value"
        assert cache.get("missing_key") == "fresh_value"  # Cached


class TestDeviceUtility:
    """Tests for device utility."""

    def test_get_available_device_cpu(self):
        """Test CPU device always works."""
        device = get_available_device("cpu")
        assert device == "cpu"

    def test_get_available_device_cuda_fallback(self):
        """Test CUDA falls back to CPU if unavailable."""
        device = get_available_device("cuda")
        assert device in ["cpu", "cuda"]  # Either is valid

    def test_get_available_device_auto_detect(self):
        """Test auto-detect returns valid device."""
        device = get_available_device(None)
        assert device in ["cpu", "cuda"]

    def test_get_available_device_invalid_falls_back(self):
        """Test invalid device name falls back to CPU."""
        device = get_available_device("invalid_device")
        assert device == "cpu"

    def test_get_available_device_uppercase(self):
        """Test device names are case-insensitive."""
        device1 = get_available_device("CPU")
        device2 = get_available_device("CUDA")

        assert device1 == "cpu"
        assert device2 in ["cpu", "cuda"]
