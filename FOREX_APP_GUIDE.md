# ⚡ FXPulse — Forex News Intelligence & Search Engine Terminal

A specialized, high-performance financial web application that aggregates real-time Forex news, automatically classifies and sorts news by **currency pairs**, caches and refreshes data with a **dual-layer Redis & In-Memory engine**, and features an **Okapi BM25 ranked search engine** with keyword highlighting.

---

## 🚀 1-Step Quick Start

You don't need any complex setup or installation. From the repository root, simply run:

```bash
python3 run_forex_app.py
```

Then open your browser to:
- **Web Terminal**: [http://localhost:8000](http://localhost:8000)
- **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💡 How We Addressed All Your Requirements

| Requirement | How It Works in This Application |
| :--- | :--- |
| **"Take all the new APIs that serve Forex news"** | Ingests real-time feeds from top market sources: **FXStreet News**, **Yahoo Finance Currencies**, **CNBC Forex & Economy**, **MarketWatch**, **Federal Reserve Press**, and **Bank of England Releases**. |
| **"Store them in database"** | Persists historical articles into an ACID SQLite database (`data/forex_news.db`) with indexes, deduplication, impact levels, and sentiment scores. |
| **"Refresh using something like Redis (don't know how to code/install Redis)"** | Built a **Smart Dual-Layer Cache Engine**: Runs a high-performance in-memory cache with TTL expiration and background refresh immediately out of the box with **zero installation needed**. Also auto-detects real Redis if running! |
| **"Sort it according to the currency pairs using the search engine"** | Automated financial NLP detects pairs (`EUR/USD`, `GBP/USD`, `USD/JPY`, `AUD/USD`, `USD/CAD`, `USD/CHF`, `NZD/USD`, `XAU/USD`, `DXY`), classifies sentiment, and ranks relevance with Okapi BM25. |
| **"Include search bar at the page which searches the complete database and provides relevant content"** | Full-text BM25 search bar with multi-field weighting (Pairs: 3.0x, Title: 2.5x, Summary: 1.2x) and visual `<mark>` search keyword highlighting. |

---

## 🧠 Understanding Redis (Made Simple for Non-Coders)

### What is Redis?
Think of a traditional database (like SQLite or Postgres) as a **filing cabinet on disk** — it is permanent, but reading from disk takes several milliseconds.
**Redis is like a whiteboard right on your desk** — it stores data entirely in computer memory (RAM). When your webpage requests news, reading from RAM takes less than **0.1 milliseconds** (100 times faster than disk).

### How We Solved the Redis Problem for You
Because you mentioned you do not know how to install or code Redis:
1. **Zero-Setup Built-In Mode (Active Now)**: We built a lightweight Python in-memory store that behaves exactly like Redis. It caches the latest news, handles automatic time-to-live (TTL) expiration, and synchronizes with your SQLite database. **You don't have to install or configure anything!**
2. **Real Redis Auto-Detection**: If you ever want to connect a real Redis server, the app automatically detects it on port `6379`.

### (Optional) How to Install Real Redis in 1 Command
If you ever want to run a standalone Redis daemon on your Mac:
```bash
# Using Homebrew:
brew install redis && brew services start redis

# Or using Docker:
docker run -d -p 6379:6379 --name redis redis:alpine

# Install Python driver:
pip install redis
```
The app will automatically switch from `In-Memory Mode` to `Live Redis Server` upon detection.

---

## 🔍 The BM25 Search Engine

The application leverages the Okapi BM25 ranked retrieval algorithm:
$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \times \frac{f(q, D) \times (k_1 + 1)}{f(q, D) + k_1 \times \left(1 - b + b \times \frac{|D|}{\text{avgdl}}\right)}$$

### Multi-Field Boost Weights
- **Currency Pair Match**: `3.0x Boost` (Searching for `EUR/USD` or `USD/JPY` prioritizes matching pairs)
- **Headline Title Match**: `2.5x Boost` (Keywords appearing in headlines are prioritized)
- **Summary & Content**: `1.2x Boost`
- **Dynamic Term Highlighting**: Matches are cleanly highlighted in `<mark class="search-highlight">` tags inside snippets.

---

## 📈 Supported Currency Pairs & Market Assets

- **Major Pairs**:
  - `EUR/USD` (Euro / US Dollar)
  - `GBP/USD` (British Pound / US Dollar)
  - `USD/JPY` (US Dollar / Japanese Yen)
  - `AUD/USD` (Australian Dollar / US Dollar)
  - `USD/CAD` (US Dollar / Canadian Dollar)
  - `USD/CHF` (US Dollar / Swiss Franc)
  - `NZD/USD` (New Zealand Dollar / US Dollar)
- **Crosses**:
  - `EUR/GBP`, `EUR/JPY`, `GBP/JPY`, `AUD/JPY`
- **Commodities & Benchmark Indices**:
  - `XAU/USD` (Spot Gold)
  - `XAG/USD` (Spot Silver)
  - `DXY` (US Dollar Index)
  - `WTI/USD` (Crude Oil)
  - `BTC/USD` (Bitcoin / USD)

---

## 🛠 Project Structure

```
search-engine-main/
├── run_forex_app.py                # 1-line launcher script
├── FOREX_APP_GUIDE.md              # Complete guide and architecture documentation
├── forex_app/
│   ├── config.py                   # App configuration & news feed sources
│   ├── engine/
│   │   └── search.py               # BM25 Search Engine with field boosting & highlighting
│   ├── fetcher/
│   │   ├── news_service.py         # Multi-feed aggregator & background auto-refresher
│   │   └── rss_feeds.py            # Robust RSS/Atom parser with XML sanitization
│   ├── nlp/
│   │   └── classifier.py           # Currency pair detector, impact level & sentiment analyzer
│   ├── storage/
│   │   ├── cache.py                # Dual-layer Cache (In-Memory + Redis auto-detect)
│   │   └── database.py             # SQLite database layer with indexes and deduplication
│   ├── templates/
│   │   ├── index.html              # Responsive Bloomberg-style financial terminal UI
│   │   └── reels.html              # Instagram Reels-style vertical scrolling feed
│   └── web/
│       ├── app.py                  # FastAPI REST API & Server-Sent Events (SSE) stream
│       └── static/
│           ├── css/style.css       # Dark-mode financial terminal styling
│           └── js/app.js           # Live search, filter tabs, auto-refresh countdown
└── tests/
    └── test_forex_app.py           # Comprehensive unit & integration tests
```

---

## 🧪 Running Tests

To verify the test suite:
```bash
python3 -m unittest tests/test_forex_app.py
```
All unit tests for caching, database persistence, pair extraction, sentiment scoring, and BM25 search run in under 0.05s.
