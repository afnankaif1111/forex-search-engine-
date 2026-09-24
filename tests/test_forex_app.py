"""Comprehensive test suite for the Forex News Search & Cache application."""

import os
import shutil
import tempfile
import unittest

from forex_app.engine.search import ForexSearchEngine, tokenize_forex
from forex_app.nlp.classifier import (
    classify_impact,
    classify_sentiment,
    detect_currency_pairs,
    extract_tags,
)
from forex_app.storage.cache import InMemoryCache, CacheManager
from forex_app.storage.database import DatabaseManager


class TestForexApp(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_forex.db")
        self.db = DatabaseManager(self.db_path)
        self.search_engine = ForexSearchEngine()
        self.cache = InMemoryCache()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ---------------- Cache Tests ----------------

    def test_in_memory_cache_crud(self):
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertTrue(self.cache.exists("key1"))

        self.cache.delete("key1")
        self.assertIsNone(self.cache.get("key1"))
        self.assertFalse(self.cache.exists("key1"))

    def test_cache_ttl_expiration(self):
        # Set with 1-second TTL
        self.cache.set("short_lived", "data", ex=1)
        self.assertEqual(self.cache.get("short_lived"), "data")
        # Artificially trigger expiration in the data structure
        self.cache._expires["short_lived"] = 0
        self.assertIsNone(self.cache.get("short_lived"))

    def test_cache_manager_status(self):
        cm = CacheManager()
        status = cm.get_status()
        self.assertIn("engine", status)
        self.assertIn("connection_status", status)
        self.assertIn("install_instructions", status)

    # ---------------- Database Tests ----------------

    def test_database_insert_and_filter(self):
        article = {
            "guid": "test-guid-1",
            "title": "Fed Chair Powell Signals Rate Cut Timing",
            "summary": "Powell noted inflation progress during speech.",
            "content": "Full commentary on FOMC interest rate decision.",
            "url": "https://example.com/powell",
            "source": "FXStreet",
            "published_at": "2026-09-24T12:00:00",
            "currency_pairs": ["EUR/USD", "DXY"],
            "primary_pair": "EUR/USD",
            "sentiment": "BEARISH",
            "sentiment_score": -0.4,
            "impact": "HIGH",
            "tags": ["Central Banks", "Interest Rates"],
        }
        inserted = self.db.insert_article(article)
        self.assertTrue(inserted)

        # Duplicate check
        duplicate = self.db.insert_article(article)
        self.assertFalse(duplicate)

        # Query by pair
        res_pair = self.db.get_articles(pair="EUR/USD")
        self.assertEqual(len(res_pair), 1)
        self.assertEqual(res_pair[0]["guid"], "test-guid-1")

        # Query by impact
        res_impact = self.db.get_articles(impact="HIGH")
        self.assertEqual(len(res_impact), 1)

        # Query by non-matching impact
        res_low = self.db.get_articles(impact="LOW")
        self.assertEqual(len(res_low), 0)

        # Stats
        stats = self.db.get_stats()
        self.assertEqual(stats["total_articles"], 1)
        self.assertEqual(stats["sentiment_breakdown"].get("BEARISH"), 1)

    # ---------------- NLP Classification Tests ----------------

    def test_pair_detection(self):
        pairs, primary = detect_currency_pairs(
            "EUR/USD plunges as ECB prepares rate cut",
            "The euro fell against the dollar following comments by Lagarde."
        )
        self.assertIn("EUR/USD", pairs)
        self.assertEqual(primary, "EUR/USD")

        # Yen & Intervention
        pairs_jpy, primary_jpy = detect_currency_pairs(
            "USD/JPY hits 155 as Bank of Japan holds",
            "Ministry of Finance warns against yen weakness."
        )
        self.assertIn("USD/JPY", pairs_jpy)
        self.assertEqual(primary_jpy, "USD/JPY")

        # Gold
        pairs_gold, _ = detect_currency_pairs(
            "Gold hits record highs as safe haven demand surges",
            "XAU/USD breached resistance."
        )
        self.assertIn("XAU/USD", pairs_gold)

    def test_impact_classification(self):
        impact_high = classify_impact(
            "US CPI and FOMC interest rate decision due today",
            "Markets brace for volatile inflation print."
        )
        self.assertEqual(impact_high, "HIGH")

        impact_med = classify_impact(
            "UK Manufacturing PMI shows modest expansion",
            "Survey shows index rose to 51.2."
        )
        self.assertEqual(impact_med, "MEDIUM")

        impact_low = classify_impact(
            "EUR/USD Technical Analysis: Consolidation continues",
            "Price hovering around 20-day moving average."
        )
        self.assertEqual(impact_low, "LOW")

    def test_sentiment_classification(self):
        sentiment, score = classify_sentiment(
            "Dollar rallies and surges to fresh highs after blockbuster jobs report",
            "Greenback strengthened across the board with strong gains."
        )
        self.assertEqual(sentiment, "BULLISH")
        self.assertGreater(score, 0)

        sentiment_bear, score_bear = classify_sentiment(
            "Euro slumps and dives as recession fears mount",
            "Disappointing economic contraction sparks heavy selloff."
        )
        self.assertEqual(sentiment_bear, "BEARISH")
        self.assertLess(score_bear, 0)

    # ---------------- Search Engine Tests ----------------

    def test_search_engine_bm25(self):
        doc1 = {
            "id": 1,
            "title": "Fed Powell discusses US inflation and interest rate policy",
            "summary": "Federal Reserve Chairman Jerome Powell answered questions on sticky inflation.",
            "currency_pairs": ["EUR/USD", "DXY"],
            "primary_pair": "EUR/USD",
            "sentiment": "NEUTRAL",
            "impact": "HIGH",
            "tags": ["Central Banks"],
        }
        doc2 = {
            "id": 2,
            "title": "Bank of Japan considers rate hikes as Yen weakens",
            "summary": "Kazuo Ueda signaled further monetary tightening if Tokyo inflation persists.",
            "currency_pairs": ["USD/JPY"],
            "primary_pair": "USD/JPY",
            "sentiment": "BULLISH",
            "impact": "HIGH",
            "tags": ["Central Banks"],
        }

        self.search_engine.add_document(doc1)
        self.search_engine.add_document(doc2)

        # Search for "Powell"
        results, hits = self.search_engine.search(query="Powell")
        self.assertEqual(hits, 1)
        self.assertEqual(results[0]["id"], 1)
        self.assertIn("<mark", results[0]["highlighted_snippet"])

        # Search for "Yen Ueda"
        results_jpy, hits_jpy = self.search_engine.search(query="Yen Ueda")
        self.assertEqual(hits_jpy, 1)
        self.assertEqual(results_jpy[0]["id"], 2)

        # Filter by pair in search
        results_filtered, hits_filtered = self.search_engine.search(query="rate", pair="USD/JPY")
        self.assertEqual(hits_filtered, 1)
        self.assertEqual(results_filtered[0]["id"], 2)


if __name__ == "__main__":
    unittest.main()
