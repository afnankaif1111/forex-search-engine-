"""Tests for the concurrent feed fetcher and the insert-id fix."""

import os
import shutil
import tempfile
import time
import unittest

import httpx

from forex_app.fetcher.async_fetch import FeedFetcher
from forex_app.storage.database import DatabaseManager

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Fed holds rates, EUR/USD rallies</title><link>https://x/1</link>
<description>Powell</description><guid>g1</guid></item>
</channel></rss>"""

FEEDS = [{"name": f"feed{i}", "url": f"https://example.test/{i}"} for i in range(5)]


class TestFeedFetcher(unittest.TestCase):

    def test_feeds_fetched_in_parallel(self):
        # Each feed takes 0.3s; sequential would be 1.5s.
        async def handler(request):
            import asyncio
            await asyncio.sleep(0.3)
            return httpx.Response(200, content=RSS)

        fetcher = FeedFetcher(transport=httpx.MockTransport(handler))
        t0 = time.perf_counter()
        results = fetcher.fetch_all(FEEDS)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 0.9)
        self.assertEqual([r["feed"] for r in results], [f["name"] for f in FEEDS])
        self.assertTrue(all(len(r["items"]) == 1 for r in results))

    def test_etag_round_trip_gives_not_modified(self):
        seen = []

        def handler(request):
            seen.append(request.headers.get("if-none-match"))
            if request.headers.get("if-none-match") == '"v1"':
                return httpx.Response(304)
            return httpx.Response(200, content=RSS, headers={"ETag": '"v1"'})

        fetcher = FeedFetcher(transport=httpx.MockTransport(handler))
        first = fetcher.fetch_all(FEEDS[:1])[0]
        second = fetcher.fetch_all(FEEDS[:1])[0]

        self.assertEqual(seen, [None, '"v1"'])
        self.assertFalse(first["not_modified"])
        self.assertEqual(len(first["items"]), 1)
        self.assertTrue(second["not_modified"])
        self.assertEqual(second["items"], [])

    def test_one_failing_feed_does_not_break_others(self):
        def handler(request):
            if request.url.path == "/0":
                return httpx.Response(500)
            if request.url.path == "/1":
                raise httpx.ConnectTimeout("slow")
            return httpx.Response(200, content=RSS)

        results = FeedFetcher(transport=httpx.MockTransport(handler)).fetch_all(FEEDS)
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[1]["status"], "error")
        self.assertTrue(all(r["status"] == "ok" for r in results[2:]))

    def test_rate_limited_feed_is_skipped_until_retry_after(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(429, headers={"Retry-After": "120"})

        fetcher = FeedFetcher(transport=httpx.MockTransport(handler))
        first = fetcher.fetch_all(FEEDS[:1])[0]
        second = fetcher.fetch_all(FEEDS[:1])[0]

        self.assertEqual(first["error"], "HTTP 429")
        self.assertEqual(second["status"], "backoff")
        self.assertEqual(calls, ["/0"])


class TestInsertReturnsId(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db = DatabaseManager(os.path.join(self.temp_dir, "t.db"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_insert_sets_row_id(self):
        a = {"guid": "a", "title": "A", "source": "s", "published_at": "2026-01-01"}
        b = {"guid": "b", "title": "B", "source": "s", "published_at": "2026-01-01"}
        self.assertTrue(self.db.insert_article(a))
        self.assertTrue(self.db.insert_article(b))
        self.assertEqual(b["id"], a["id"] + 1)

        dup = dict(a)
        dup.pop("id")
        self.assertFalse(self.db.insert_article(dup))
        self.assertNotIn("id", dup)


if __name__ == "__main__":
    unittest.main()
