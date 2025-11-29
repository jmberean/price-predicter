"""Simple caching utility for data connectors with LRU eviction and TTL support."""

import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)


class SimpleCache:
    """
    Simple LRU cache with TTL (time-to-live) support.

    Features:
    - LRU eviction when max_size is reached
    - TTL-based expiration
    - Thread-safe (basic)
    - Cache statistics (hits/misses)
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key not in self._cache:
            self._misses += 1
            return None

        value, timestamp = self._cache[key]

        # Check if expired
        if time.time() - timestamp > self.ttl_seconds:
            logger.debug(f"Cache entry expired: {key}")
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (mark as recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # If key exists, remove it first (will be re-added at end)
        if key in self._cache:
            del self._cache[key]

        # Add new entry
        self._cache[key] = (value, time.time())

        # Evict oldest if over max_size
        if len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            logger.debug(f"Cache full, evicting oldest entry: {oldest_key}")
            del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, size
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size,
        }


def make_cache_key(asset_id: str, start: datetime, end: datetime, **kwargs) -> str:
    """
    Create cache key for market data.

    Args:
        asset_id: Asset identifier
        start: Start datetime
        end: End datetime

    Returns:
        Cache key string
    """
    return f"{asset_id}:{start.isoformat()}:{end.isoformat()}"


def cached_fetch(
    cache: SimpleCache,
    key: str,
    fetch_fn: Callable[[], Any],
) -> Any:
    """
    Fetch with caching - get from cache or call fetch function.

    Args:
        cache: Cache instance
        key: Cache key
        fetch_fn: Function to call if cache miss (should return value to cache)

    Returns:
        Cached or freshly fetched value
    """
    # Try cache first
    cached_value = cache.get(key)
    if cached_value is not None:
        logger.debug(f"Cache hit: {key}")
        return cached_value

    # Cache miss - fetch and store
    logger.debug(f"Cache miss: {key}")
    value = fetch_fn()
    cache.set(key, value)
    return value
