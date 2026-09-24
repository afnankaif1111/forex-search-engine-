"""BM25 Search Engine for Forex News.

Implements Okapi BM25 ranked retrieval with field-boosting (Title: 2.5x, Pairs: 3.0x, Summary: 1.2x),
exact phrase matches, currency pair filtering, and snippet generation with keyword highlighting.
"""

from collections import defaultdict
import heapq
import math
import re
import threading
from typing import Any, Optional


def tokenize_forex(text: str) -> list[str]:
    """Tokenize financial text, preserving currency pairs, symbols, and numbers."""
    if not text:
        return []
    # Lowercase
    cleaned = text.lower()

    # Extract currency pairs like eur/usd -> also add eurusd
    pair_matches = re.findall(r"([a-z]{3})/([a-z]{3})", cleaned)
    extra_tokens = []
    for c1, c2 in pair_matches:
        extra_tokens.append(f"{c1}/{c2}")
        extra_tokens.append(f"{c1}{c2}")

    # Standard alphanumeric words
    words = re.findall(r"[a-z0-9]+", cleaned)
    return words + extra_tokens


class ForexSearchEngine:
    """Fast in-memory BM25 search engine with field weighting and dynamic updates."""

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()

        # Inverted index: term -> list[(doc_id, tf)]
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        # Doc lengths: doc_id -> total token count
        self.doc_lens: dict[int, int] = {}
        # Document metadata store: doc_id -> doc dict
        self.docs: dict[int, dict[str, Any]] = {}

        self.total_tokens = 0
        self.n_docs = 0

    def add_document(self, doc: dict[str, Any]) -> None:
        """Add or update a document in the index."""
        doc_id = doc["id"]
        title = doc.get("title", "")
        summary = doc.get("summary", "")
        pairs = doc.get("currency_pairs", [])
        if isinstance(pairs, list):
            pairs_text = " ".join(pairs)
        else:
            pairs_text = str(pairs)
        primary_pair = doc.get("primary_pair", "")
        tags = doc.get("tags", [])
        tags_text = " ".join(tags) if isinstance(tags, list) else str(tags)
        content = doc.get("content", "")

        # Tokenize fields
        title_toks = tokenize_forex(title)
        pairs_toks = tokenize_forex(f"{pairs_text} {primary_pair}")
        summary_toks = tokenize_forex(summary)
        body_toks = tokenize_forex(content)
        tags_toks = tokenize_forex(tags_text)

        with self._lock:
            # If doc already exists, remove it first
            if doc_id in self.docs:
                self.remove_document(doc_id)

            # Field-weighted term frequency
            tf: dict[str, float] = defaultdict(float)

            for t in title_toks:
                tf[t] += 2.5  # Title boost
            for t in pairs_toks:
                tf[t] += 3.0  # Currency Pair boost
            for t in summary_toks:
                tf[t] += 1.2  # Summary boost
            for t in tags_toks:
                tf[t] += 1.5  # Tags boost
            for t in body_toks:
                tf[t] += 1.0  # Body

            doc_len = len(title_toks) + len(summary_toks) + len(body_toks) + len(pairs_toks)
            self.doc_lens[doc_id] = doc_len
            self.docs[doc_id] = doc
            self.n_docs += 1
            self.total_tokens += doc_len

            # Insert into postings
            for term, weight in tf.items():
                self.postings[term].append((doc_id, weight))

    def remove_document(self, doc_id: int) -> None:
        """Remove document from postings and lengths."""
        with self._lock:
            if doc_id not in self.docs:
                return
            old_len = self.doc_lens.pop(doc_id, 0)
            self.docs.pop(doc_id, None)
            self.n_docs = max(0, self.n_docs - 1)
            self.total_tokens = max(0, self.total_tokens - old_len)

            # Filter out doc_id from postings
            for term in list(self.postings.keys()):
                self.postings[term] = [p for p in self.postings[term] if p[0] != doc_id]
                if not self.postings[term]:
                    del self.postings[term]

    def build_from_db(self, db_manager) -> int:
        """Rebuild complete index from SQLite database."""
        articles = db_manager.get_all_for_indexing()
        with self._lock:
            self.postings.clear()
            self.doc_lens.clear()
            self.docs.clear()
            self.total_tokens = 0
            self.n_docs = 0

            for art in articles:
                self.add_document(art)
        return self.n_docs

    def search(
        self,
        query: str = "",
        pair: Optional[str] = None,
        sentiment: Optional[str] = None,
        impact: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Ranked BM25 search with metadata filtering and highlight snippets."""
        with self._lock:
            if self.n_docs == 0:
                return [], 0

            query_terms = tokenize_forex(query)
            avgdl = (self.total_tokens / self.n_docs) if self.n_docs > 0 else 1.0

            # Candidate filter set (pair, sentiment, impact)
            candidate_ids: set[int] = set(self.docs.keys())

            if pair and pair.upper() != "ALL":
                pair_norm = pair.strip().upper()
                filtered = set()
                for doc_id in candidate_ids:
                    doc = self.docs[doc_id]
                    p_pairs = doc.get("currency_pairs", [])
                    primary = doc.get("primary_pair", "")
                    if primary == pair_norm or (isinstance(p_pairs, list) and pair_norm in p_pairs):
                        filtered.add(doc_id)
                candidate_ids = filtered

            if sentiment and sentiment.upper() != "ALL":
                sent_norm = sentiment.strip().upper()
                candidate_ids = {
                    d_id for d_id in candidate_ids
                    if self.docs[d_id].get("sentiment", "").upper() == sent_norm
                }

            if impact and impact.upper() != "ALL":
                imp_norm = impact.strip().upper()
                candidate_ids = {
                    d_id for d_id in candidate_ids
                    if self.docs[d_id].get("impact", "").upper() == imp_norm
                }

            if not candidate_ids:
                return [], 0

            # If no query string, rank purely by ID descending (recency)
            if not query_terms:
                sorted_ids = sorted(candidate_ids, reverse=True)
                total_hits = len(sorted_ids)
                page_ids = sorted_ids[offset : offset + limit]
                results = []
                for d_id in page_ids:
                    doc = dict(self.docs[d_id])
                    doc["relevance_score"] = 1.0
                    doc["highlighted_snippet"] = doc.get("summary") or doc.get("title", "")
                    results.append(doc)
                return results, total_hits

            # BM25 Scoring
            scores: dict[int, float] = defaultdict(float)
            unique_query_terms = set(query_terms)

            for t in unique_query_terms:
                plist = self.postings.get(t)
                if not plist:
                    continue

                df = len(plist)
                # BM25 standard IDF
                idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

                for doc_id, tf in plist:
                    if doc_id not in candidate_ids:
                        continue
                    dl = self.doc_lens.get(doc_id, avgdl)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                    term_score = idf * (tf * (self.k1 + 1.0)) / denom
                    scores[doc_id] += term_score

            if not scores:
                return [], 0

            # Sort by score descending
            ranked = heapq.nlargest(len(scores), scores.items(), key=lambda kv: kv[1])
            total_hits = len(ranked)
            page_ranked = ranked[offset : offset + limit]

            results = []
            for doc_id, score in page_ranked:
                doc = dict(self.docs[doc_id])
                doc["relevance_score"] = round(score, 3)
                doc["highlighted_snippet"] = self._generate_snippet(doc, unique_query_terms)
                results.append(doc)

            return results, total_hits

    def _generate_snippet(self, doc: dict[str, Any], query_terms: set[str]) -> str:
        """Create a summary snippet with highlighted search terms."""
        text = doc.get("summary") or doc.get("title") or ""
        if not text or not query_terms:
            return text

        # Sort terms by length descending to prevent substring collisions
        sorted_terms = sorted(query_terms, key=len, reverse=True)
        highlighted = text
        for term in sorted_terms:
            if len(term) < 2:
                continue
            pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
            highlighted = pattern.sub(r"<mark class='search-highlight'>\1</mark>", highlighted)

        return highlighted


# Singleton instance
search_engine = ForexSearchEngine()
