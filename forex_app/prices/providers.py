"""Price provider adapters.

Every provider turns one upstream feed into calls of
``emit(symbol, price, ts_ms, ref, source)`` where ``symbol`` is canonical
(see assets.py) and ``ref`` is the reference price used for % change
(24h open for crypto, previous close for everything else; None if the
message does not carry one).

Push providers (WebSockets) reconnect with exponential backoff. Message
parsing lives in small pure functions so it is testable without a network.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from typing import Any, Callable, Optional

import httpx
import websockets

logger = logging.getLogger(__name__)

Emit = Callable[[str, float, int, Optional[float], str], None]


class ProviderDown(Exception):
    """Raised when a provider failed too many times in a row to keep trying."""


class WebSocketProvider:
    name = "ws"

    def __init__(self, assets: list[dict[str, Any]], max_failures: int | None = None):
        self.assets = assets
        self.max_failures = max_failures  # None = retry forever
        self.connected = False
        self.ticks = 0

    def url(self) -> str:
        raise NotImplementedError

    async def on_open(self, ws) -> None:
        pass

    def parse(self, raw: str) -> list[tuple[str, float, int, Optional[float]]]:
        raise NotImplementedError

    async def run(self, emit: Emit) -> None:
        failures = 0
        delay = 1.0
        while True:
            try:
                async with websockets.connect(self.url(), open_timeout=8,
                                              ping_interval=20, max_size=2**22) as ws:
                    await self.on_open(ws)
                    self.connected = True
                    logger.info("%s connected (%d assets)", self.name, len(self.assets))
                    failures, delay = 0, 1.0
                    async for raw in ws:
                        for sym, price, ts, ref in self.parse(raw):
                            self.ticks += 1
                            emit(sym, price, ts, ref, self.name)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("%s disconnected: %s: %s", self.name, type(e).__name__, e)
            finally:
                self.connected = False
            failures += 1
            if self.max_failures is not None and failures >= self.max_failures:
                raise ProviderDown(self.name)
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 30.0)


class BinanceProvider(WebSocketProvider):
    """Every aggregated trade (true real-time) + 1s mini-ticker for 24h open."""
    name = "binance"

    def __init__(self, assets, **kw):
        super().__init__([a for a in assets if a.get("binance")], **kw)
        self._by_stream = {a["binance"]: a["symbol"] for a in self.assets}

    def url(self) -> str:
        streams = "/".join(f"{a['binance'].lower()}@{kind}"
                           for a in self.assets for kind in ("aggTrade", "miniTicker"))
        return f"wss://stream.binance.com:9443/stream?streams={streams}"

    def parse(self, raw):
        d = json.loads(raw).get("data", {})
        sym = self._by_stream.get(d.get("s"))
        if not sym:
            return []
        if d.get("e") == "aggTrade":
            return [(sym, float(d["p"]), int(d["T"]), None)]
        if d.get("e") == "24hrMiniTicker":
            return [(sym, float(d["c"]), int(d["E"]), float(d["o"]))]
        return []


class CoinbaseProvider(WebSocketProvider):
    name = "coinbase"

    def __init__(self, assets, **kw):
        super().__init__([a for a in assets if a.get("coinbase")], **kw)
        self._by_product = {a["coinbase"]: a["symbol"] for a in self.assets}

    def url(self) -> str:
        return "wss://ws-feed.exchange.coinbase.com"

    async def on_open(self, ws) -> None:
        await ws.send(json.dumps({"type": "subscribe", "channels": ["ticker"],
                                  "product_ids": list(self._by_product)}))

    def parse(self, raw):
        d = json.loads(raw)
        sym = self._by_product.get(d.get("product_id"))
        if d.get("type") != "ticker" or not sym:
            return []
        ts = int(datetime.fromisoformat(d["time"].replace("Z", "+00:00")).timestamp() * 1000)
        ref = float(d["open_24h"]) if d.get("open_24h") else None
        return [(sym, float(d["price"]), ts, ref)]


class FinnhubProvider(WebSocketProvider):
    """US stocks + OANDA forex trades. Needs a free API key."""
    name = "finnhub"

    def __init__(self, assets, api_key: str, **kw):
        super().__init__([a for a in assets if a.get("finnhub")], **kw)
        self.api_key = api_key
        self._by_code = {a["finnhub"]: a["symbol"] for a in self.assets}

    def url(self) -> str:
        return f"wss://ws.finnhub.io?token={self.api_key}"

    async def on_open(self, ws) -> None:
        for code in self._by_code:
            await ws.send(json.dumps({"type": "subscribe", "symbol": code}))

    def parse(self, raw):
        d = json.loads(raw)
        if d.get("type") != "trade":
            return []
        # One message batches many trades; keep only the latest per symbol.
        latest: dict[str, tuple[float, int]] = {}
        for t in d.get("data", []):
            sym = self._by_code.get(t.get("s"))
            if sym and (sym not in latest or t["t"] >= latest[sym][1]):
                latest[sym] = (float(t["p"]), int(t["t"]))
        return [(s, p, ts, None) for s, (p, ts) in latest.items()]


class FailoverProvider:
    """Runs providers in order, moving to the next one when one is down."""

    def __init__(self, providers: list[WebSocketProvider]):
        self.providers = providers
        self.name = "+".join(p.name for p in providers)
        self.active = providers[0].name

    @property
    def connected(self) -> bool:
        return any(p.connected for p in self.providers)

    @property
    def ticks(self) -> int:
        return sum(p.ticks for p in self.providers)

    async def run(self, emit: Emit) -> None:
        i = 0
        while True:
            p = self.providers[i % len(self.providers)]
            self.active = p.name
            try:
                await p.run(emit)
            except ProviderDown:
                logger.warning("%s down, failing over", p.name)
                i += 1


class YahooPoller:
    """Polls Yahoo's chart endpoint for every asset concurrently.

    Not a push feed, so latency is the poll interval. Used for forex/stocks
    when no Finnhub key is configured, for assets Finnhub can't stream
    (indices, oil), and for previous-close reference prices.

    Unofficial endpoint: may change or rate-limit without notice.
    """
    name = "yahoo"
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}"

    def __init__(self, assets, interval: float = 3.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.assets = [a for a in assets if a.get("yahoo")]
        self.interval = interval
        self.transport = transport
        self.connected = False
        self.ticks = 0
        self.last_poll_ms = 0.0

    @staticmethod
    def parse(payload: dict) -> Optional[tuple[float, int, Optional[float]]]:
        try:
            m = payload["chart"]["result"][0]["meta"]
            price = float(m["regularMarketPrice"])
            ts = int(m["regularMarketTime"]) * 1000
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        ref = m.get("chartPreviousClose") or m.get("previousClose")
        return price, ts, float(ref) if ref else None

    async def poll_once(self, client: httpx.AsyncClient, emit: Emit) -> int:
        t0 = time.perf_counter()

        async def one(a):
            r = await client.get(self.URL.format(a["yahoo"]),
                                 params={"interval": "1d", "range": "1d"})
            if r.status_code != 200:
                return a, None
            return a, self.parse(r.json())

        results = await asyncio.gather(*(one(a) for a in self.assets),
                                       return_exceptions=True)
        n = 0
        for res in results:
            if isinstance(res, Exception) or res[1] is None:
                continue
            a, (price, ts, ref) = res
            emit(a["symbol"], price, ts, ref, self.name)
            n += 1
        self.ticks += n
        self.connected = n > 0
        self.last_poll_ms = round((time.perf_counter() - t0) * 1000, 1)
        return n

    async def run(self, emit: Emit) -> None:
        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"},
                                     timeout=6.0, transport=self.transport) as client:
            while True:
                try:
                    await self.poll_once(client, emit)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.connected = False
                    logger.warning("yahoo poll failed: %s", e)
                await asyncio.sleep(self.interval)
