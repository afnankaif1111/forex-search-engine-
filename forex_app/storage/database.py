"""SQLite database layer for storing and querying Forex news articles."""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Any, Optional

from ..config import DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe SQLite database manager for Forex news storage."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")  # High concurrency
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guid TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                content TEXT,
                url TEXT,
                source TEXT NOT NULL,
                published_at TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                currency_pairs TEXT NOT NULL,
                primary_pair TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                sentiment_score REAL DEFAULT 0.0,
                impact TEXT NOT NULL,
                tags TEXT DEFAULT '[]'
            );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published_at DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_primary_pair ON articles(primary_pair);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON articles(source);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_impact ON articles(impact);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON articles(sentiment);")

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS refresh_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_count INTEGER DEFAULT 0,
                new_count INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                details TEXT
            );
            """)

            conn.commit()
            conn.close()

    def clear_all(self) -> None:
        """Purge all articles and refresh logs from database."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM articles")
            cursor.execute("DELETE FROM refresh_logs")
            conn.commit()
            conn.close()

    def insert_article(self, article: dict[str, Any]) -> bool:
        """Insert a single article. Returns True if inserted, False if duplicate."""
        now_str = datetime.utcnow().isoformat()
        guid = article.get("guid") or article.get("url") or article.get("title", "")
        pairs = article.get("currency_pairs", [])
        if isinstance(pairs, list):
            pairs_json = json.dumps(pairs)
        else:
            pairs_json = str(pairs)

        tags = article.get("tags", [])
        tags_json = json.dumps(tags) if isinstance(tags, list) else str(tags)

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                INSERT OR IGNORE INTO articles (
                    guid, title, summary, content, url, source,
                    published_at, fetched_at, currency_pairs, primary_pair,
                    sentiment, sentiment_score, impact, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    guid,
                    article.get("title", "").strip(),
                    article.get("summary", "").strip(),
                    article.get("content", "").strip(),
                    article.get("url", "").strip(),
                    article.get("source", "Forex News").strip(),
                    article.get("published_at", now_str),
                    article.get("fetched_at", now_str),
                    pairs_json,
                    article.get("primary_pair", "GENERAL"),
                    article.get("sentiment", "NEUTRAL"),
                    float(article.get("sentiment_score", 0.0)),
                    article.get("impact", "MEDIUM"),
                    tags_json,
                ))
                inserted = cursor.rowcount > 0
                conn.commit()
                return inserted
            except Exception as e:
                logger.error("Error inserting article: %s", e)
                return False
            finally:
                conn.close()

    def insert_many(self, articles: list[dict[str, Any]]) -> int:
        """Bulk insert articles. Returns count of newly inserted items."""
        new_count = 0
        for item in articles:
            if self.insert_article(item):
                new_count += 1
        return new_count

    def log_refresh(self, source: str, fetched: int, new_items: int, status: str, details: str = "") -> None:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO refresh_logs (timestamp, source, fetched_count, new_count, status, details)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.utcnow().isoformat(), source, fetched, new_items, status, details))
            conn.commit()
            conn.close()

    def get_articles(
        self,
        limit: int = 50,
        offset: int = 0,
        pair: Optional[str] = None,
        sentiment: Optional[str] = None,
        impact: Optional[str] = None,
        source: Optional[str] = None,
        ids: Optional[list[int]] = None,
        sort_by: str = "published_at",
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch articles with optional filters."""
        query = "SELECT * FROM articles WHERE 1=1"
        params: list[Any] = []

        if ids is not None:
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            query += f" AND id IN ({placeholders})"
            params.extend(ids)

        if pair and pair.upper() != "ALL":
            # Check either primary pair or inside JSON currency_pairs
            query += " AND (primary_pair = ? OR currency_pairs LIKE ?)"
            params.append(pair)
            params.append(f'%"{pair}"%')

        if sentiment and sentiment.upper() != "ALL":
            query += " AND sentiment = ?"
            params.append(sentiment.upper())

        if impact and impact.upper() != "ALL":
            query += " AND impact = ?"
            params.append(impact.upper())

        if source and source.upper() != "ALL":
            query += " AND source = ?"
            params.append(source)

        allowed_sorts = {"published_at", "id", "sentiment_score", "impact"}
        sort_col = sort_by if sort_by in allowed_sorts else "published_at"
        direction = "ASC" if ascending else "DESC"

        # If ids were provided in a specific order (e.g. from BM25 search relevance score),
        # keep search relevance ordering unless requested otherwise
        if ids and sort_by == "relevance":
            # order by CASE
            case_stmt = " CASE id " + " ".join(f"WHEN {doc_id} THEN {idx}" for idx, doc_id in enumerate(ids)) + " END"
            query += f" ORDER BY {case_stmt}"
        else:
            query += f" ORDER BY {sort_col} {direction}"

        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            d = dict(r)
            try:
                d["currency_pairs"] = json.loads(d["currency_pairs"])
            except Exception:
                d["currency_pairs"] = [d["primary_pair"]]
            try:
                d["tags"] = json.loads(d.get("tags") or "[]")
            except Exception:
                d["tags"] = []
            results.append(d)
        return results

    def get_article_by_id(self, article_id: int) -> Optional[dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        try:
            d["currency_pairs"] = json.loads(d["currency_pairs"])
        except Exception:
            d["currency_pairs"] = [d["primary_pair"]]
        return d

    def get_all_for_indexing(self) -> list[dict[str, Any]]:
        """Retrieve minimal article data for BM25 search indexing."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, summary, content, currency_pairs, primary_pair, source FROM articles")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_pair_counts(self) -> dict[str, int]:
        """Count articles associated with each currency pair."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT primary_pair, COUNT(*) as c FROM articles GROUP BY primary_pair")
        rows = cursor.fetchall()
        conn.close()
        return {r["primary_pair"]: r["c"] for r in rows}

    def get_stats(self) -> dict[str, Any]:
        """Aggregate stats for dashboard sidebar."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM articles")
        total_articles = cursor.fetchone()["total"]

        cursor.execute("SELECT sentiment, COUNT(*) as cnt FROM articles GROUP BY sentiment")
        sentiment_breakdown = {r["sentiment"]: r["cnt"] for r in cursor.fetchall()}

        cursor.execute("SELECT impact, COUNT(*) as cnt FROM articles GROUP BY impact")
        impact_breakdown = {r["impact"]: r["cnt"] for r in cursor.fetchall()}

        cursor.execute("SELECT source, COUNT(*) as cnt FROM articles GROUP BY source ORDER BY cnt DESC LIMIT 5")
        top_sources = {r["source"]: r["cnt"] for r in cursor.fetchall()}

        cursor.execute("SELECT MAX(published_at) as latest FROM articles")
        latest_pub = cursor.fetchone()["latest"]

        cursor.execute("SELECT MAX(timestamp) as last_refreshed FROM refresh_logs")
        last_refreshed_row = cursor.fetchone()
        last_refreshed = last_refreshed_row["last_refreshed"] if last_refreshed_row else None

        conn.close()

        return {
            "total_articles": total_articles,
            "sentiment_breakdown": sentiment_breakdown,
            "impact_breakdown": impact_breakdown,
            "top_sources": top_sources,
            "latest_published_at": latest_pub,
            "last_refreshed": last_refreshed,
        }


# Singleton instance
db = DatabaseManager()
