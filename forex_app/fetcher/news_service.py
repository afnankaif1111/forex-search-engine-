"""News Aggregation and Background Refresh Service.

Coordinates RSS fetching, NLP classification, SQLite database persistence,
in-memory / Redis caching, and BM25 search index synchronization.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable

from ..config import DEFAULT_REFRESH_INTERVAL, RSS_FEEDS
from ..engine.search import search_engine
from ..nlp.classifier import (
    classify_impact,
    classify_sentiment,
    detect_currency_pairs,
    extract_tags,
)
from ..storage.cache import cache
from ..storage.database import db
from .rss_feeds import fetch_rss_feed

logger = logging.getLogger(__name__)


class NewsService:
    """Central service managing Forex news ingestion, caching, and background refreshes."""

    def __init__(self, refresh_interval: int = DEFAULT_REFRESH_INTERVAL):
        self.refresh_interval = refresh_interval
        self.is_running = False
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._subscribers: list[asyncio.Queue] = []
        self._sub_lock = threading.Lock()
        self.last_refresh_time: float = 0.0
        self.last_refresh_stats: dict[str, Any] = {}

    def initialize(self) -> None:
        """Initialize database, fetch real API feeds if empty, build search index, and prime cache."""
        stats = db.get_stats()
        if stats["total_articles"] == 0:
            logger.info("Database is empty. Fetching live news from APIs & RSS feeds...")
            self.fetch_all_sources()

        # Build BM25 search index
        doc_count = search_engine.build_from_db(db)
        logger.info("BM25 search engine indexed %d documents.", doc_count)

        # Prime the cache
        self._update_cache()

    def start_background_worker(self) -> None:
        """Launch background auto-refresh thread."""
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Background news refresher started (interval: %ds).", self.refresh_interval)

    def stop_background_worker(self) -> None:
        """Stop background worker."""
        if not self.is_running:
            return
        self._stop_event.set()
        self.is_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=3.0)
        logger.info("Background news refresher stopped.")

    def _refresh_loop(self) -> None:
        """Periodic loop to fetch new articles."""
        while not self._stop_event.is_set():
            # Wait for next interval or stop event
            if self._stop_event.wait(timeout=self.refresh_interval):
                break
            try:
                logger.info("Executing scheduled news refresh...")
                self.fetch_all_sources()
            except Exception as e:
                logger.error("Error during scheduled refresh: %s", e)

    def _process_and_store(self, raw_article: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Run NLP classification, insert to SQLite, and index in BM25 search engine."""
        title = raw_article.get("title", "")
        summary = raw_article.get("summary", "")
        content = raw_article.get("content", "") or summary

        # 1. Detect Currency Pairs
        pairs, primary_pair = detect_currency_pairs(title, f"{summary} {content}")

        # 2. Classify Impact
        impact = classify_impact(title, f"{summary} {content}")

        # 3. Classify Sentiment
        sentiment, sentiment_score = classify_sentiment(title, f"{summary} {content}")

        # 4. Extract Tags
        tags = extract_tags(title, f"{summary} {content}")

        article_record = {
            "guid": raw_article.get("guid") or raw_article.get("url") or title,
            "title": title,
            "summary": summary,
            "content": content,
            "url": raw_article.get("url", ""),
            "source": raw_article.get("source", "Forex News"),
            "published_at": raw_article.get("published_at", datetime.utcnow().isoformat()),
            "currency_pairs": pairs,
            "primary_pair": primary_pair,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "impact": impact,
            "tags": tags,
        }

        # Store in SQLite
        inserted = db.insert_article(article_record)
        if inserted:
            # Query back inserted record to get assigned SQLite autoincrement ID
            saved = db.get_articles(limit=1, ids=None, sort_by="id", ascending=False)
            if saved:
                article_record["id"] = saved[0]["id"]
                # Add to BM25 search engine
                search_engine.add_document(article_record)

        return inserted, article_record

    def fetch_all_sources(self) -> dict[str, Any]:
        """Fetch all configured news feeds, categorize, store, and update cache."""
        total_fetched = 0
        new_inserted = 0
        feed_results = []

        for feed in RSS_FEEDS:
            name = feed["name"]
            url = feed["url"]
            try:
                items = fetch_rss_feed(name, url, timeout=6.0)
                feed_new = 0
                for item in items:
                    total_fetched += 1
                    inserted, _ = self._process_and_store(item)
                    if inserted:
                        feed_new += 1
                        new_inserted += 1

                db.log_refresh(name, len(items), feed_new, "SUCCESS")
                feed_results.append({
                    "feed": name,
                    "fetched": len(items),
                    "new": feed_new,
                    "status": "ok",
                })
            except Exception as e:
                logger.warning("Feed %s failed: %s", name, e)
                db.log_refresh(name, 0, 0, "FAILED", str(e))
                feed_results.append({
                    "feed": name,
                    "fetched": 0,
                    "new": 0,
                    "status": "error",
                    "error": str(e),
                })

        self.last_refresh_time = time.time()
        self.last_refresh_stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_fetched": total_fetched,
            "new_inserted": new_inserted,
            "feeds": feed_results,
        }

        # Update cache & notify subscribers if new articles arrived
        self._update_cache()

        if new_inserted > 0:
            self._notify_subscribers({
                "type": "new_articles",
                "count": new_inserted,
                "timestamp": datetime.utcnow().isoformat(),
            })

        return self.last_refresh_stats

    def _update_cache(self) -> None:
        """Update Redis/In-Memory cache keys for ultra-fast reading."""
        # Top 50 latest articles
        latest = db.get_articles(limit=50)
        cache.set_json("forex:latest_articles", latest, ex=600)

        # Pair breakdown counts
        pair_counts = db.get_pair_counts()
        cache.set_json("forex:pair_counts", pair_counts, ex=600)

        # Overall Stats
        stats = db.get_stats()
        cache.set_json("forex:stats", stats, ex=600)

    # ---------------- SSE Real-Time Streaming ----------------
    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE client queue."""
        q = asyncio.Queue()
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Unregister an SSE client queue."""
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _notify_subscribers(self, message: dict[str, Any]) -> None:
        """Broadcast message to all active SSE queues."""
        with self._sub_lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(message)
                except Exception:
                    dead.append(q)
            for d in dead:
                self._subscribers.remove(d)


# Singleton instance
news_service = NewsService()
