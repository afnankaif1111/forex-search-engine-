"""Tracked market assets and their symbol on each price provider.

The canonical symbol (e.g. "EUR/USD", "AAPL", "BTC/USD") is what the API,
the UI and the news classifier use; each provider has its own spelling.
"""

from typing import Any


def _fx(sym: str, yahoo: str) -> dict[str, Any]:
    base, quote = sym.split("/")
    return {"symbol": sym, "class": "forex", "yahoo": yahoo,
            "finnhub": f"OANDA:{base}_{quote}"}


def _stock(sym: str, name: str) -> dict[str, Any]:
    return {"symbol": sym, "class": "stock", "name": name, "yahoo": sym,
            "finnhub": sym}


def _crypto(base: str) -> dict[str, Any]:
    return {"symbol": f"{base}/USD", "class": "crypto",
            "binance": f"{base}USDT", "coinbase": f"{base}-USD"}


ASSETS: list[dict[str, Any]] = [
    # Forex
    _fx("EUR/USD", "EURUSD=X"),
    _fx("GBP/USD", "GBPUSD=X"),
    _fx("USD/JPY", "JPY=X"),
    _fx("AUD/USD", "AUDUSD=X"),
    _fx("USD/CAD", "CAD=X"),
    _fx("USD/CHF", "CHF=X"),
    _fx("NZD/USD", "NZDUSD=X"),
    _fx("EUR/GBP", "EURGBP=X"),
    _fx("EUR/JPY", "EURJPY=X"),
    _fx("GBP/JPY", "GBPJPY=X"),
    # Commodities & indices (Yahoo futures / index symbols; no Finnhub free stream)
    {"symbol": "XAU/USD", "class": "commodity", "name": "Gold", "yahoo": "GC=F",
     "finnhub": "OANDA:XAU_USD"},
    {"symbol": "XAG/USD", "class": "commodity", "name": "Silver", "yahoo": "SI=F",
     "finnhub": "OANDA:XAG_USD"},
    {"symbol": "WTI/USD", "class": "commodity", "name": "Crude Oil", "yahoo": "CL=F"},
    {"symbol": "DXY", "class": "index", "name": "US Dollar Index", "yahoo": "DX-Y.NYB"},
    {"symbol": "SPX", "class": "index", "name": "S&P 500", "yahoo": "^GSPC"},
    # Stocks
    _stock("AAPL", "Apple"),
    _stock("MSFT", "Microsoft"),
    _stock("NVDA", "Nvidia"),
    _stock("TSLA", "Tesla"),
    _stock("AMZN", "Amazon"),
    _stock("GOOGL", "Alphabet"),
    _stock("META", "Meta"),
    _stock("SPY", "S&P 500 ETF"),
    _stock("QQQ", "Nasdaq 100 ETF"),
    # Crypto
    _crypto("BTC"),
    _crypto("ETH"),
    _crypto("SOL"),
    _crypto("XRP"),
    _crypto("DOGE"),
    _crypto("ADA"),
]

BY_SYMBOL: dict[str, dict[str, Any]] = {a["symbol"]: a for a in ASSETS}


def normalize_symbol(raw: str) -> str | None:
    """Map user input like 'eurusd', 'btc', 'aapl' to a canonical symbol."""
    s = raw.strip().upper().replace("-", "/").replace("_", "/")
    if s in BY_SYMBOL:
        return s
    if len(s) == 6 and "/" not in s and f"{s[:3]}/{s[3:]}" in BY_SYMBOL:
        return f"{s[:3]}/{s[3:]}"
    if f"{s}/USD" in BY_SYMBOL:
        return f"{s}/USD"
    return None
