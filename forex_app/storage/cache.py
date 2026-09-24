"""Cache Manager: Seamless Dual-Layer Caching Engine.

Supports both:
1. True Redis Server (via redis-py if installed and running)
2. High-Performance Built-in In-Memory Cache (pure Python, zero dependencies, thread-safe, TTL support)

Provides a 100% drop-in Redis-compatible interface so users who do NOT have Redis
installed can run immediately with zero friction!
"""

import fnmatch
import json
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Thread-safe in-memory cache supporting TTL, hashes, and lists with Redis-compatible semantics."""

    def __init__(self):
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._expires: dict[str, float] = {}
        self._created_at = time.time()

        # Background reaper for expired keys
        self._stop_reaper = False
        self._reaper_thread = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper_thread.start()

    def _is_expired(self, key: str) -> bool:
        if key in self._expires:
            if time.time() > self._expires[key]:
                self._del_internal(key)
                return True
        return False

    def _del_internal(self, key: str) -> None:
        self._data.pop(key, None)
        self._expires.pop(key, None)

    def _reap_loop(self):
        while not self._stop_reaper:
            time.sleep(30)
            now = time.time()
            with self._lock:
                expired_keys = [k for k, exp in self._expires.items() if now > exp]
                for k in expired_keys:
                    self._del_internal(k)

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if self._is_expired(key):
                return None
            val = self._data.get(key)
            if val is None:
                return None
            return str(val) if not isinstance(val, str) else val

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        with self._lock:
            self._data[key] = value
            if ex is not None:
                self._expires[key] = time.time() + ex
            else:
                self._expires.pop(key, None)
            return True

    def setex(self, key: str, time_seconds: int, value: Any) -> bool:
        return self.set(key, value, ex=time_seconds)

    def delete(self, *keys: str) -> int:
        count = 0
        with self._lock:
            for k in keys:
                if k in self._data:
                    self._del_internal(k)
                    count += 1
        return count

    def exists(self, key: str) -> bool:
        with self._lock:
            if self._is_expired(key):
                return False
            return key in self._data

    def keys(self, pattern: str = "*") -> list[str]:
        with self._lock:
            valid_keys = [k for k in list(self._data.keys()) if not self._is_expired(k)]
            if pattern == "*":
                return valid_keys
            return [k for k in valid_keys if fnmatch.fnmatch(k, pattern)]

    def flushall(self) -> bool:
        with self._lock:
            self._data.clear()
            self._expires.clear()
            return True

    # Hashes
    def hset(self, name: str, key: str, value: Any) -> int:
        with self._lock:
            if self._is_expired(name):
                self._data[name] = {}
            if name not in self._data or not isinstance(self._data[name], dict):
                self._data[name] = {}
            is_new = key not in self._data[name]
            self._data[name][key] = str(value)
            return 1 if is_new else 0

    def hget(self, name: str, key: str) -> Optional[str]:
        with self._lock:
            if self._is_expired(name):
                return None
            h = self._data.get(name)
            if isinstance(h, dict):
                return h.get(key)
            return None

    def hgetall(self, name: str) -> dict[str, str]:
        with self._lock:
            if self._is_expired(name):
                return {}
            h = self._data.get(name)
            if isinstance(h, dict):
                return dict(h)
            return {}

    # Lists
    def lpush(self, name: str, *values: Any) -> int:
        with self._lock:
            if self._is_expired(name):
                self._data[name] = []
            if name not in self._data or not isinstance(self._data[name], list):
                self._data[name] = []
            for v in values:
                self._data[name].insert(0, str(v))
            return len(self._data[name])

    def lrange(self, name: str, start: int, end: int) -> list[str]:
        with self._lock:
            if self._is_expired(name):
                return []
            lst = self._data.get(name)
            if isinstance(lst, list):
                if end == -1:
                    return list(lst[start:])
                return list(lst[start : end + 1])
            return []

    def ltrim(self, name: str, start: int, end: int) -> bool:
        with self._lock:
            if self._is_expired(name):
                return True
            lst = self._data.get(name)
            if isinstance(lst, list):
                if end == -1:
                    self._data[name] = lst[start:]
                else:
                    self._data[name] = lst[start : end + 1]
            return True

    def key_count(self) -> int:
        with self._lock:
            return len([k for k in self._data if not self._is_expired(k)])


class CacheManager:
    """Manages Redis connection with automatic fallback to InMemoryCache."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.is_redis_connected = False
        self.connection_error: Optional[str] = None
        self.memory_cache = InMemoryCache()
        self.start_time = time.time()

        self._try_connect_redis()

    def _try_connect_redis(self) -> bool:
        """Attempt to connect to Redis server."""
        try:
            import redis

            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
            # Ping to verify active daemon
            client.ping()
            self.redis_client = client
            self.is_redis_connected = True
            self.connection_error = None
            logger.info("Successfully connected to live Redis server at %s", self.redis_url)
            return True
        except ImportError:
            self.redis_client = None
            self.is_redis_connected = False
            self.connection_error = (
                "Python 'redis' package is not installed. Using built-in high-speed memory cache."
            )
            return False
        except Exception as e:
            self.redis_client = None
            self.is_redis_connected = False
            self.connection_error = f"Cannot reach Redis server at {self.redis_url} ({e})."
            return False

    def get(self, key: str) -> Optional[str]:
        if self.is_redis_connected and self.redis_client:
            try:
                return self.redis_client.get(key)
            except Exception as e:
                logger.warning("Redis get error, falling back to memory: %s", e)
        return self.memory_cache.get(key)

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        # Always update memory cache as backup
        self.memory_cache.set(key, val_str, ex=ex)

        if self.is_redis_connected and self.redis_client:
            try:
                if ex:
                    return bool(self.redis_client.setex(key, ex, val_str))
                return bool(self.redis_client.set(key, val_str))
            except Exception as e:
                logger.warning("Redis set error: %s", e)
        return True

    def setex(self, key: str, seconds: int, value: Any) -> bool:
        return self.set(key, value, ex=seconds)

    def get_json(self, key: str) -> Optional[Any]:
        val = self.get(key)
        if val is None:
            return None
        try:
            return json.loads(val)
        except Exception:
            return val

    def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        return self.set(key, json.dumps(value), ex=ex)

    def delete(self, *keys: str) -> int:
        count = self.memory_cache.delete(*keys)
        if self.is_redis_connected and self.redis_client:
            try:
                return self.redis_client.delete(*keys)
            except Exception:
                pass
        return count

    def exists(self, key: str) -> bool:
        if self.is_redis_connected and self.redis_client:
            try:
                return bool(self.redis_client.exists(key))
            except Exception:
                pass
        return self.memory_cache.exists(key)

    def keys(self, pattern: str = "*") -> list[str]:
        if self.is_redis_connected and self.redis_client:
            try:
                return self.redis_client.keys(pattern)
            except Exception:
                pass
        return self.memory_cache.keys(pattern)

    def flushall(self) -> bool:
        self.memory_cache.flushall()
        if self.is_redis_connected and self.redis_client:
            try:
                return bool(self.redis_client.flushall())
            except Exception:
                pass
        return True

    def lpush(self, name: str, *values: Any) -> int:
        self.memory_cache.lpush(name, *values)
        if self.is_redis_connected and self.redis_client:
            try:
                return self.redis_client.lpush(name, *values)
            except Exception:
                pass
        return 1

    def lrange(self, name: str, start: int, end: int) -> list[str]:
        if self.is_redis_connected and self.redis_client:
            try:
                return self.redis_client.lrange(name, start, end)
            except Exception:
                pass
        return self.memory_cache.lrange(name, start, end)

    def ltrim(self, name: str, start: int, end: int) -> bool:
        self.memory_cache.ltrim(name, start, end)
        if self.is_redis_connected and self.redis_client:
            try:
                return bool(self.redis_client.ltrim(name, start, end))
            except Exception:
                pass
        return True

    def get_status(self) -> dict[str, Any]:
        """Returns comprehensive diagnostic info for the UI and APIs."""
        # Try reconnecting if not connected
        if not self.is_redis_connected:
            self._try_connect_redis()

        uptime = int(time.time() - self.start_time)
        return {
            "is_redis": self.is_redis_connected,
            "engine": "Redis Server" if self.is_redis_connected else "Built-in High-Speed Memory Engine",
            "redis_url": self.redis_url if self.is_redis_connected else None,
            "connection_status": "Connected" if self.is_redis_connected else "Operating in Built-in Mode",
            "connection_note": (
                "Connected to real Redis daemon on port 6379."
                if self.is_redis_connected
                else "Running smoothly on built-in thread-safe memory store. Zero install required!"
            ),
            "keys_count": self.memory_cache.key_count(),
            "uptime_seconds": uptime,
            "install_instructions": {
                "mac_brew": "brew install redis && brew services start redis",
                "docker": "docker run -d -p 6379:6379 --name redis redis:alpine",
                "python_driver": "pip install redis",
                "explanation": (
                    "Redis is an in-memory key-value database commonly used to cache market data "
                    "with sub-millisecond latency. You do NOT need to install it: this application "
                    "includes a built-in Redis emulation layer that works instantly out of the box."
                ),
            },
        }


# Singleton instance
cache = CacheManager()
