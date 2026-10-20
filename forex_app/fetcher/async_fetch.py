"""Concurrent feed fetcher with HTTP conditional requests.

All feeds are downloaded in parallel, so a refresh costs as long as the
slowest feed rather than the sum of all of them. Each feed's ETag and
Last-Modified headers are remembered and sent back on the next refresh; a
304 Not Modified response skips both the download and the parse. A feed that
answers 429/503 is skipped until its Retry-After (default 5 minutes) passes.
"""

import asyncio
import logging
import ssl
import threading
import time
from typing import Any

import httpx

from .rss_feeds import DEFAULT_HEADERS, parse_feed

logger = logging.getLogger(__name__)

DEFAULT_BACKOFF_SECONDS = 300.0


class FeedFetcher:
    """Fetches many feeds at once and remembers their validators between runs."""

    def __init__(self, timeout: float = 4.0, max_connections: int = 32,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.timeout = timeout
        self.transport = transport  # injectable for tests
        self.max_connections = max_connections
        # url -> {"etag": ..., "last_modified": ...}
        self._validators: dict[str, dict[str, str]] = {}
        # url -> monotonic time before which the feed is not requested
        self._backoff_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def _conditional_headers(self, url: str, extra: dict[str, str]) -> dict[str, str]:
        headers = {**DEFAULT_HEADERS, **extra}
        with self._lock:
            v = self._validators.get(url, {})
        if v.get("etag"):
            headers["If-None-Match"] = v["etag"]
        if v.get("last_modified"):
            headers["If-Modified-Since"] = v["last_modified"]
        return headers

    def _remember(self, url: str, response: httpx.Response) -> None:
        v = {}
        if response.headers.get("etag"):
            v["etag"] = response.headers["etag"]
        if response.headers.get("last-modified"):
            v["last_modified"] = response.headers["last-modified"]
        if v:
            with self._lock:
                self._validators[url] = v

    def _back_off(self, url: str, response: httpx.Response) -> None:
        try:
            delay = float(response.headers.get("retry-after", ""))
        except ValueError:
            delay = DEFAULT_BACKOFF_SECONDS
        with self._lock:
            self._backoff_until[url] = time.monotonic() + min(delay, 3600.0)

    async def _get(self, client: httpx.AsyncClient, feed: dict[str, Any]) -> httpx.Response:
        url = feed["url"]
        return await client.get(url, headers=self._conditional_headers(url, feed.get("headers", {})))

    async def _fetch_one(self, client: httpx.AsyncClient, insecure: httpx.AsyncClient,
                         feed: dict[str, Any]) -> dict[str, Any]:
        name, url = feed["name"], feed["url"]
        t0 = time.perf_counter()
        result: dict[str, Any] = {"feed": name, "items": [], "status": "ok",
                                  "not_modified": False}
        with self._lock:
            until = self._backoff_until.get(url, 0.0)
        if time.monotonic() < until:
            result["status"] = "backoff"
            result["error"] = f"rate-limited, retry in {until - time.monotonic():.0f}s"
            result["elapsed_ms"] = 0.0
            return result
        try:
            try:
                response = await self._get(client, feed)
            except httpx.ConnectError as e:
                # Some government/central-bank feeds serve incomplete
                # certificate chains; retry that one feed without verification.
                if not isinstance(e.__cause__, ssl.SSLError) and "SSL" not in str(e):
                    raise
                response = await self._get(insecure, feed)

            if response.status_code == 304:
                result["not_modified"] = True
            elif response.status_code == 200:
                self._remember(url, response)
                result["items"] = parse_feed(name, response.content)
            else:
                if response.status_code in (429, 503):
                    self._back_off(url, response)
                result["status"] = "error"
                result["error"] = f"HTTP {response.status_code}"
        except Exception as e:
            result["status"] = "error"
            result["error"] = f"{type(e).__name__}: {e}"
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    async def fetch_all_async(self, feeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limits = httpx.Limits(max_connections=self.max_connections)
        timeout = httpx.Timeout(self.timeout)
        opts = dict(timeout=timeout, limits=limits, follow_redirects=True,
                    transport=self.transport)
        async with httpx.AsyncClient(**opts) as client, \
                httpx.AsyncClient(**opts, verify=False) as insecure:
            return await asyncio.gather(
                *(self._fetch_one(client, insecure, f) for f in feeds))

    def fetch_all(self, feeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Blocking wrapper; call from a worker thread, not from the event loop."""
        return asyncio.run(self.fetch_all_async(feeds))
