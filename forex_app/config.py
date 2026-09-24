import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = str(DATA_DIR / "forex_news.db")
CACHE_DIR = str(DATA_DIR / "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Redis Configuration (Auto-detected; falls back to in-memory store if Redis is unavailable)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Background Refresh Interval (seconds)
DEFAULT_REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL_SECONDS", "120"))

# Web Server Settings
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", "8000"))

# Standard Forex Pairs
MAJOR_PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CAD",
    "USD/CHF",
    "NZD/USD",
]

MINOR_PAIRS = [
    "EUR/GBP",
    "EUR/JPY",
    "GBP/JPY",
    "AUD/JPY",
    "EUR/CHF",
    "GBP/AUD",
]

COMMODITY_CRYPTO_PAIRS = [
    "XAU/USD",  # Gold
    "XAG/USD",  # Silver
    "DXY",      # US Dollar Index
    "WTI/USD",  # Crude Oil
    "BTC/USD",  # Bitcoin / USD
]

ALL_PAIRS = MAJOR_PAIRS + MINOR_PAIRS + COMMODITY_CRYPTO_PAIRS

# Real-Time Forex & Financial RSS Feeds
RSS_FEEDS = [
    {
        "name": "FXStreet News",
        "url": "https://www.fxstreet.com/rss/news",
        "category": "forex",
        "reliability": 0.95,
    },
    {
        "name": "Yahoo Finance Currencies",
        "url": "https://finance.yahoo.com/news/rssindex",
        "category": "forex",
        "reliability": 0.90,
    },
    {
        "name": "CNBC Forex & Economy",
        "url": "https://search.cnbc.com/rs/search/view.html?partnerId=2000&keywords=forex&sort=date&output=rss",
        "category": "economy",
        "reliability": 0.90,
    },
    {
        "name": "MarketWatch Realtime",
        "url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
        "category": "markets",
        "reliability": 0.85,
    },
    {
        "name": "Federal Reserve Press",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "category": "central_bank",
        "reliability": 1.0,
    },
    {
        "name": "Bank of England Releases",
        "url": "https://www.bankofengland.co.uk/rss/news",
        "category": "central_bank",
        "reliability": 0.95,
    },
]

# API Keys (Optional - only if user provides them via UI or environment)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
