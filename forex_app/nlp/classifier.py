"""Financial NLP module for currency pair recognition, impact assessment, and sentiment analysis."""

import re
from typing import Optional

# Standard Pair Normalization Map
PAIR_SYNONYMS = {
    # Majors
    "EURUSD": "EUR/USD",
    "EUR/USD": "EUR/USD",
    "EUR-USD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "GBP/USD": "GBP/USD",
    "GBP-USD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USD/JPY": "USD/JPY",
    "USD-JPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "AUD/USD": "AUD/USD",
    "AUD-USD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "USD/CAD": "USD/CAD",
    "USD-CAD": "USD/CAD",
    "USDCHF": "USD/CHF",
    "USD/CHF": "USD/CHF",
    "USD-CHF": "USD/CHF",
    "NZDUSD": "NZD/USD",
    "NZD/USD": "NZD/USD",
    "NZD-USD": "NZD/USD",
    # Minors & Crosses
    "EURGBP": "EUR/GBP",
    "EUR/GBP": "EUR/GBP",
    "EURJPY": "EUR/JPY",
    "EUR/JPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "GBP/JPY": "GBP/JPY",
    "AUDJPY": "AUD/JPY",
    "AUD/JPY": "AUD/JPY",
    "EURCHF": "EUR/CHF",
    "EUR/CHF": "EUR/CHF",
    # Commodities & Indices
    "XAUUSD": "XAU/USD",
    "XAU/USD": "XAU/USD",
    "GOLD": "XAU/USD",
    "XAGUSD": "XAG/USD",
    "XAG/USD": "XAG/USD",
    "SILVER": "XAG/USD",
    "DXY": "DXY",
    "DOLLAR INDEX": "DXY",
    "US DOLLAR INDEX": "DXY",
    "CRUDE OIL": "WTI/USD",
    "WTI": "WTI/USD",
    "BRENT": "WTI/USD",
    "BTCUSD": "BTC/USD",
    "BTC/USD": "BTC/USD",
    "BITCOIN": "BTC/USD",
}

# Currency indicators and central banks
CURRENCY_KEYWORDS = {
    "USD": [
        "federal reserve", "fed", "powell", "fomc", "greenback", "us dollar", "treasury",
        "wall street", "non-farm", "nfp", "us cpi", "us inflation"
    ],
    "EUR": [
        "european central bank", "ecb", "lagarde", "eurozone", "euro", "bund", "german", "france"
    ],
    "GBP": [
        "bank of england", "boe", "bailey", "pound", "sterling", "uk cpi", "gilts", "britain", "uk economy"
    ],
    "JPY": [
        "bank of japan", "boj", "ueda", "yen", "japanese", "jgb", "tokyo", "intervention"
    ],
    "AUD": [
        "reserve bank of australia", "rba", "aussie", "australian dollar", "bullock"
    ],
    "CAD": [
        "bank of canada", "boc", "loonie", "canadian dollar", "macklem"
    ],
    "CHF": [
        "swiss national bank", "snb", "swiss franc", "switzerland", "jordan"
    ],
    "NZD": [
        "reserve bank of new zealand", "rbnz", "kiwi", "new zealand dollar", "orr"
    ],
    "XAU": [
        "gold", "bullion", "yellow metal", "xau"
    ],
}

# Impact Lexicons
HIGH_IMPACT_KEYWORDS = [
    "interest rate", "rate decision", "rate hike", "rate cut", "cpi", "inflation",
    "fomc", "non-farm payroll", "nfp", "gdp", "central bank", "powell", "lagarde",
    "emergency meeting", "intervention", "breaking", "monetary policy statement",
    "core pce", "unemployment rate", "geopolitical crisis"
]

MEDIUM_IMPACT_KEYWORDS = [
    "pmi", "retail sales", "ppi", "consumer confidence", "housing starts",
    "trade balance", "durable goods", "minutes", "speech", "forecast", "outlook",
    "current account", "industrial production", "crude inventories"
]

# Sentiment Lexicons
BULLISH_WORDS = {
    "rally": 2.0, "surge": 2.0, "jump": 1.5, "gain": 1.0, "gains": 1.0, "highs": 1.5,
    "soar": 2.0, "hawkish": 2.0, "breakout": 1.5, "strong": 1.2, "expansion": 1.2,
    "upgrade": 1.5, "optimism": 1.2, "rebound": 1.5, "advance": 1.0, "advances": 1.0,
    "climb": 1.2, "strengthen": 1.5, "bull": 1.5, "bulls": 1.5, "bullish": 2.0,
    "outperform": 1.5, "beat": 1.2, "positive": 1.0, "recovery": 1.2
}

BEARISH_WORDS = {
    "slump": 2.0, "drop": 1.5, "drops": 1.5, "plunge": 2.0, "dive": 1.8, "lows": 1.5,
    "tumble": 2.0, "dovish": 2.0, "recession": 2.0, "weak": 1.5, "contraction": 1.5,
    "downgrade": 1.5, "loss": 1.0, "losses": 1.2, "selloff": 2.0, "fears": 1.5,
    "slide": 1.5, "decline": 1.2, "retreat": 1.2, "falter": 1.5, "bear": 1.5,
    "bears": 1.5, "bearish": 2.0, "sink": 1.8, "underperform": 1.5, "negative": 1.0,
    "collapse": 2.0, "warning": 1.2
}


def detect_currency_pairs(title: str, text: str) -> tuple[list[str], str]:
    """Identify all currency pairs mentioned and determine the primary pair."""
    full_text = f"{title} {text}".upper()
    title_upper = title.upper()
    found_pairs: set[str] = set()

    # 1. Check direct pair symbols (e.g. EUR/USD, GBPUSD, XAU/USD, DXY)
    for pattern, canonical in PAIR_SYNONYMS.items():
        # Match as discrete token or standard pattern
        token_regex = r"(?:\b|\s|^)" + re.escape(pattern) + r"(?:\b|\s|$)"
        if re.search(token_regex, full_text):
            found_pairs.add(canonical)

    # 2. Check individual currency names/central banks
    detected_currencies: set[str] = set()
    for curr, keywords in CURRENCY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", full_text, re.IGNORECASE):
                detected_currencies.add(curr)
                break

    # Synthesize standard pairs if currencies detected together
    if "EUR" in detected_currencies and "USD" in detected_currencies:
        found_pairs.add("EUR/USD")
    if "GBP" in detected_currencies and "USD" in detected_currencies:
        found_pairs.add("GBP/USD")
    if "JPY" in detected_currencies and "USD" in detected_currencies:
        found_pairs.add("USD/JPY")
    if "AUD" in detected_currencies and "USD" in detected_currencies:
        found_pairs.add("AUD/USD")
    if "CAD" in detected_currencies and "USD" in detected_currencies:
        found_pairs.add("USD/CAD")
    if "CHF" in detected_currencies and "USD" in detected_currencies:
        found_pairs.add("USD/CHF")
    if "NZD" in detected_currencies and "USD" in detected_currencies:
        found_pairs.add("NZD/USD")
    if "EUR" in detected_currencies and "GBP" in detected_currencies:
        found_pairs.add("EUR/GBP")
    if "EUR" in detected_currencies and "JPY" in detected_currencies:
        found_pairs.add("EUR/JPY")
    if "GBP" in detected_currencies and "JPY" in detected_currencies:
        found_pairs.add("GBP/JPY")
    if "XAU" in detected_currencies:
        found_pairs.add("XAU/USD")

    # If only one major currency was found without a pair, link to its USD benchmark
    if not found_pairs:
        if "USD" in detected_currencies:
            found_pairs.add("DXY")
        elif "EUR" in detected_currencies:
            found_pairs.add("EUR/USD")
        elif "GBP" in detected_currencies:
            found_pairs.add("GBP/USD")
        elif "JPY" in detected_currencies:
            found_pairs.add("USD/JPY")
        elif "AUD" in detected_currencies:
            found_pairs.add("AUD/USD")
        elif "CAD" in detected_currencies:
            found_pairs.add("USD/CAD")
        else:
            found_pairs.add("GENERAL")

    # Determine primary pair: check title matches first
    pairs_list = list(found_pairs)
    primary = pairs_list[0]
    for p in pairs_list:
        if p in title_upper or p.replace("/", "") in title_upper:
            primary = p
            break

    return pairs_list, primary


def classify_impact(title: str, text: str) -> str:
    """Classify market impact as HIGH, MEDIUM, or LOW."""
    content = f"{title} {text}".lower()

    for kw in HIGH_IMPACT_KEYWORDS:
        if kw in content:
            return "HIGH"

    for kw in MEDIUM_IMPACT_KEYWORDS:
        if kw in content:
            return "MEDIUM"

    return "LOW"


def classify_sentiment(title: str, text: str) -> tuple[str, float]:
    """Calculate sentiment score (-1.0 to +1.0) and label (BULLISH, BEARISH, NEUTRAL)."""
    title_words = re.findall(r"[a-z0-9]+", title.lower())
    text_words = re.findall(r"[a-z0-9]+", text.lower())

    bull_score = 0.0
    bear_score = 0.0

    # Title is weighted 2.5x more heavily than body text
    for w in title_words:
        if w in BULLISH_WORDS:
            bull_score += BULLISH_WORDS[w] * 2.5
        if w in BEARISH_WORDS:
            bear_score += BEARISH_WORDS[w] * 2.5

    for w in text_words:
        if w in BULLISH_WORDS:
            bull_score += BULLISH_WORDS[w] * 1.0
        if w in BEARISH_WORDS:
            bear_score += BEARISH_WORDS[w] * 1.0

    total = bull_score + bear_score
    if total == 0:
        return "NEUTRAL", 0.0

    net_score = (bull_score - bear_score) / (total + 2.0)
    net_score = max(-1.0, min(1.0, round(net_score, 2)))

    if net_score >= 0.15:
        return "BULLISH", net_score
    elif net_score <= -0.15:
        return "BEARISH", net_score
    else:
        return "NEUTRAL", net_score


def extract_tags(title: str, text: str) -> list[str]:
    """Extract topical tags like Central Bank, Inflation, Technicals, etc."""
    content = f"{title} {text}".lower()
    tags = []

    if any(k in content for k in ["fed", "ecb", "boj", "boe", "rba", "central bank", "fomc"]):
        tags.append("Central Banks")
    if any(k in content for k in ["cpi", "inflation", "price index", "cost of living"]):
        tags.append("Inflation")
    if any(k in content for k in ["interest rate", "rate cut", "rate hike", "monetary policy"]):
        tags.append("Interest Rates")
    if any(k in content for k in ["nfp", "payrolls", "jobs", "unemployment", "labor market"]):
        tags.append("Employment")
    if any(k in content for k in ["technical", "support", "resistance", "breakout", "moving average", "chart"]):
        tags.append("Technical Analysis")
    if any(k in content for k in ["gold", "xau", "silver", "crude", "oil", "commodity"]):
        tags.append("Commodities")
    if any(k in content for k in ["geopolitics", "war", "tariff", "election", "sanctions"]):
        tags.append("Geopolitics")

    return tags if tags else ["Forex Market"]
