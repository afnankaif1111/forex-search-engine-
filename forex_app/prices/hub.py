"""Live price hub: collects ticks from all providers and fans them out.

- Keeps the latest tick per symbol in memory (O(1) quote lookups).
- Coalesces updates: subscribers get one batch every FLUSH_MS containing
  only the symbols that changed, so a burst of 500 BTC trades costs the
  browser one message, not 500.
- Rolls 1-minute OHLC bars in memory and persists finished bars to SQLite.
- Mirrors the latest ticks into the cache (Redis when available).

Run standalone to watch ticks:
    python3 -m forex_app.prices.hub --symbols BTC/USD,EUR/USD,AAPL --seconds 10
"""

import argparse
import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Optional

from .assets import ASSETS, BY_SYMBOL
from .providers import (BinanceProvider, CoinbaseProvider, FailoverProvider,
                        FinnhubProvider, YahooPoller)

logger = logging.getLogger(__name__)

FLUSH_MS = 100
BAR_MS = 60_000
BARS_IN_MEMORY = 240
QUEUE_MAX = 50


def build_providers(assets: list[dict[str, Any]], finnhub_key: str = "",
                    yahoo_interval: float = 3.0) -> list:
    providers: list = []
    crypto = [a for a in assets if a["class"] == "crypto"]
    if crypto:
        providers.append(FailoverProvider([
            BinanceProvider(crypto, max_failures=3),
            CoinbaseProvider(crypto, max_failures=3),
        ]))
    rest = [a for a in assets if a["class"] != "crypto"]
    if finnhub_key:
        providers.append(FinnhubProvider(rest, finnhub_key))
        # Yahoo still supplies previous closes and the assets Finnhub lacks;
        # stale Yahoo prices never overwrite fresher Finnhub ticks (see on_tick).
        providers.append(YahooPoller(rest, interval=30.0))
    elif rest:
        providers.append(YahooPoller(rest, interval=yahoo_interval))
    return providers


class PriceHub:
    def __init__(self, providers: Optional[list] = None, cache=None, db=None):
        self.providers = providers if providers is not None else build_providers(
            ASSETS, os.getenv("FINNHUB_API_KEY", ""),
            float(os.getenv("YAHOO_POLL_SECONDS", "3")))
        self.cache = cache
        self.db = db
        self.last: dict[str, dict[str, Any]] = {}
        self._ref: dict[str, float] = {}
        self._dirty: set[str] = set()
        self._bar: dict[str, dict[str, Any]] = {}
        self.bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=BARS_IN_MEMORY))
        self._finished_bars: list[tuple] = []
        self._subscribers: dict[asyncio.Queue, Optional[set[str]]] = {}
        self._tasks: list[asyncio.Task] = []
        self.started_at = 0.0
        self.ticks_in = 0
        self.lag_ms: dict[str, float] = {}

    # ---------------- ingest ----------------
    def on_tick(self, symbol: str, price: float, ts: int,
                ref: Optional[float], source: str) -> None:
        self.ticks_in += 1
        if ref:
            self._ref[symbol] = ref
        prev = self.last.get(symbol)
        if prev and ts < prev["ts"]:
            return  # older than what we have (e.g. Yahoo behind Finnhub)
        if prev and ts == prev["ts"] and price == prev["price"]:
            return  # unchanged (closed market polled again)
        self.lag_ms[source] = max(0.0, time.time() * 1000 - ts)
        r = self._ref.get(symbol)
        asset = BY_SYMBOL.get(symbol, {})
        self.last[symbol] = {
            "symbol": symbol,
            "class": asset.get("class", "other"),
            "price": price,
            "ref": r,
            "change_pct": round((price - r) / r * 100, 3) if r else None,
            "ts": ts,
            "source": source,
        }
        self._dirty.add(symbol)
        self._update_bar(symbol, price, ts)

    def _update_bar(self, symbol: str, price: float, ts: int) -> None:
        start = ts - ts % BAR_MS
        bar = self._bar.get(symbol)
        if bar is None or start > bar["t"]:
            if bar is not None:
                self.bars[symbol].append(bar)
                self._finished_bars.append((symbol, bar["t"], bar["o"], bar["h"],
                                            bar["l"], bar["c"], bar["n"]))
            self._bar[symbol] = {"t": start, "o": price, "h": price, "l": price,
                                 "c": price, "n": 1}
        elif start == bar["t"]:
            bar["h"] = max(bar["h"], price)
            bar["l"] = min(bar["l"], price)
            bar["c"] = price
            bar["n"] += 1

    # ---------------- fan-out ----------------
    def subscribe(self, symbols: Optional[set[str]] = None) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self._subscribers[q] = symbols
        return q

    def set_symbols(self, q: asyncio.Queue, symbols: Optional[set[str]]) -> None:
        if q in self._subscribers:
            self._subscribers[q] = symbols

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.pop(q, None)

    def snapshot(self, symbols: Optional[set[str]] = None) -> list[dict[str, Any]]:
        return [t for s, t in self.last.items() if symbols is None or s in symbols]

    def flush(self) -> int:
        if not self._dirty:
            return 0
        changed = [self.last[s] for s in self._dirty]
        self._dirty.clear()
        for q, want in list(self._subscribers.items()):
            batch = changed if want is None else [t for t in changed if t["symbol"] in want]
            if not batch:
                continue
            if q.full():  # slow client: drop its oldest batch, keep the newest
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(batch)
        if self.cache is not None:
            for t in changed:
                self.cache.set_json(f"px:{t['symbol']}", t, ex=3600)
        return len(changed)

    async def _flush_loop(self) -> None:
        last_persist = time.monotonic()
        while True:
            await asyncio.sleep(FLUSH_MS / 1000)
            self.flush()
            if self.db is not None and self._finished_bars and \
                    time.monotonic() - last_persist > 5:
                rows, self._finished_bars = self._finished_bars, []
                last_persist = time.monotonic()
                try:
                    await asyncio.to_thread(self.db.insert_bars, rows)
                except Exception as e:
                    logger.warning("persisting bars failed: %s", e)

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        if self._tasks:
            return
        self.started_at = time.time()
        for p in self.providers:
            self._tasks.append(asyncio.create_task(p.run(self.on_tick), name=p.name))
        self._tasks.append(asyncio.create_task(self._flush_loop(), name="flush"))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def get_bars(self, symbol: str) -> list[dict[str, Any]]:
        out = list(self.bars.get(symbol, []))
        if symbol in self._bar:
            out.append(dict(self._bar[symbol]))
        return out

    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._tasks),
            "symbols_live": len(self.last),
            "ticks_in": self.ticks_in,
            "subscribers": len(self._subscribers),
            "providers": [{
                "name": p.name,
                "active": getattr(p, "active", p.name),
                "connected": p.connected,
                "ticks": p.ticks,
                "lag_ms": round(self.lag_ms.get(getattr(p, "active", p.name), 0), 1),
            } for p in self.providers],
        }


async def _watch(symbols: Optional[set[str]], seconds: float) -> None:
    hub = PriceHub()
    hub.start()
    q = hub.subscribe(symbols)
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            try:
                batch = await asyncio.wait_for(q.get(), timeout=max(0.1, end - time.monotonic()))
            except asyncio.TimeoutError:
                break
            for t in batch:
                lag = time.time() * 1000 - t["ts"]
                chg = f"{t['change_pct']:+.2f}%" if t["change_pct"] is not None else "   n/a"
                print(f"{t['symbol']:9} {t['price']:>14,.5f} {chg:>8}  {t['source']:8} lag {lag:7.0f}ms")
    finally:
        print(hub.status())
        await hub.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="comma-separated, default all")
    ap.add_argument("--seconds", type=float, default=10)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    syms = {s.strip() for s in args.symbols.split(",") if s.strip()} or None
    asyncio.run(_watch(syms, args.seconds))


if __name__ == "__main__":
    main()
