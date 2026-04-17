"""
Response Cache - In-memory cache for repeat queries.

Exact-match cache: hash(query + model) -> cached response.
Configurable TTL (default 1 hour) and max size.
"""
import hashlib
import logging
import threading
import time
from typing import Optional

from liftmind.config import settings

logger = logging.getLogger(__name__)

_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()


def _make_key(query: str, lift_model: Optional[str]) -> str:
    """Generate cache key from query + model."""
    raw = f"{query.strip().lower()}|{(lift_model or '').lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached_response(query: str, lift_model: Optional[str] = None) -> Optional[dict]:
    """
    Look up a cached response.

    Returns the full result dict if found and not expired, else None.
    """
    if settings.RESPONSE_CACHE_TTL <= 0:
        return None

    key = _make_key(query, lift_model)

    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None

        # Check TTL
        if time.time() - entry["timestamp"] > settings.RESPONSE_CACHE_TTL:
            del _cache[key]
            return None

        logger.info(f"Cache hit for query: {query[:50]}...")
        return entry["result"]


def store_response(query: str, lift_model: Optional[str], result: dict) -> None:
    """
    Store a response in the cache.

    Evicts oldest entries if cache exceeds max size.
    """
    if settings.RESPONSE_CACHE_TTL <= 0:
        return

    key = _make_key(query, lift_model)

    with _cache_lock:
        # Evict oldest if at capacity
        if len(_cache) >= settings.RESPONSE_CACHE_MAX_SIZE and key not in _cache:
            oldest_key = min(_cache, key=lambda k: _cache[k]["timestamp"])
            del _cache[oldest_key]

        _cache[key] = {
            "result": result,
            "timestamp": time.time()
        }


def clear_cache() -> int:
    """Clear all cached responses. Returns number of entries cleared."""
    with _cache_lock:
        count = len(_cache)
        _cache.clear()
        return count


def cache_stats() -> dict:
    """Return cache statistics."""
    with _cache_lock:
        now = time.time()
        active = sum(1 for e in _cache.values()
                     if now - e["timestamp"] <= settings.RESPONSE_CACHE_TTL)
        return {
            "total_entries": len(_cache),
            "active_entries": active,
            "max_size": settings.RESPONSE_CACHE_MAX_SIZE,
            "ttl_seconds": settings.RESPONSE_CACHE_TTL
        }
