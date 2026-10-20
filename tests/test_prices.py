"""Tests for the live price plane: provider parsing, hub logic, API."""

import asyncio
import json
import os
import shutil
import tempfile
import unittest

import httpx

from forex_app.prices.assets import ASSETS, normalize_symbol
from forex_app.prices.hub import PriceHub, build_providers
from forex_app.prices.providers import (BinanceProvider, CoinbaseProvider,
                                        FinnhubProvider, YahooPoller)
from forex_app.storage.database import DatabaseManager

CRYPTO = [a for a in ASSETS if a["class"] == "crypto"]


class TestSymbols(unittest.TestCase):

    def test_normalize(self):
        self.assertEqual(normalize_symbol("eurusd"), "EUR/USD")
        self.assertEqual(normalize_symbol("eur-usd"), "EUR/USD")
        self.assertEqual(normalize_symbol("btc"), "BTC/USD")
        self.assertEqual(normalize_symbol("aapl"), "AAPL")
        self.assertIsNone(normalize_symbol("nope"))


class TestProviderParsing(unittest.TestCase):

    def test_binance_trade_and_ticker(self):
        p = BinanceProvider(CRYPTO)
        trade = json.dumps({"stream": "btcusdt@aggTrade",
                            "data": {"e": "aggTrade", "s": "BTCUSDT", "p": "82150.5", "T": 1700000000123}})
        tick = json.dumps({"stream": "btcusdt@miniTicker",
                           "data": {"e": "24hrMiniTicker", "s": "BTCUSDT", "c": "82151", "o": "80000", "E": 1700000000200}})
        self.assertEqual(p.parse(trade), [("BTC/USD", 82150.5, 1700000000123, None)])
        self.assertEqual(p.parse(tick), [("BTC/USD", 82151.0, 1700000000200, 80000.0)])
        self.assertIn("btcusdt@aggTrade", p.url())

    def test_coinbase_ticker(self):
        p = CoinbaseProvider(CRYPTO)
        msg = json.dumps({"type": "ticker", "product_id": "ETH-USD", "price": "2412.86",
                          "open_24h": "2426.18", "time": "2026-10-20T10:00:00.5Z"})
        [(sym, price, ts, ref)] = p.parse(msg)
        self.assertEqual((sym, price, ref), ("ETH/USD", 2412.86, 2426.18))
        self.assertEqual(ts % 1000, 500)
        self.assertEqual(p.parse(json.dumps({"type": "subscriptions"})), [])

    def test_finnhub_keeps_latest_trade_per_symbol(self):
        p = FinnhubProvider(ASSETS, api_key="x")
        msg = json.dumps({"type": "trade", "data": [
            {"s": "AAPL", "p": 100.0, "t": 2}, {"s": "AAPL", "p": 99.0, "t": 1},
            {"s": "OANDA:EUR_USD", "p": 1.12, "t": 3}]})
        self.assertEqual(sorted(p.parse(msg)),
                         [("AAPL", 100.0, 2, None), ("EUR/USD", 1.12, 3, None)])

    def test_yahoo_poll_once(self):
        def handler(request):
            if "EURUSD" in request.url.path:
                return httpx.Response(200, json={"chart": {"result": [{"meta": {
                    "regularMarketPrice": 1.1233, "regularMarketTime": 1700000000,
                    "chartPreviousClose": 1.1217}}]}})
            return httpx.Response(404)

        got = []
        poller = YahooPoller([a for a in ASSETS if a["symbol"] in ("EUR/USD", "AAPL")],
                             transport=httpx.MockTransport(handler))

        async def go():
            async with httpx.AsyncClient(transport=poller.transport) as c:
                return await poller.poll_once(c, lambda *a: got.append(a))

        self.assertEqual(asyncio.run(go()), 1)
        self.assertEqual(got, [("EUR/USD", 1.1233, 1700000000000, 1.1217, "yahoo")])

    def test_build_providers_without_and_with_finnhub(self):
        names = [p.name for p in build_providers(ASSETS)]
        self.assertEqual(names, ["binance+coinbase", "yahoo"])
        names = [p.name for p in build_providers(ASSETS, finnhub_key="k")]
        self.assertEqual(names, ["binance+coinbase", "finnhub", "yahoo"])


class TestHub(unittest.TestCase):

    def setUp(self):
        self.hub = PriceHub(providers=[])

    def test_change_pct_uses_reference(self):
        self.hub.on_tick("BTC/USD", 80000.0, 1, 79000.0, "binance")
        self.hub.on_tick("BTC/USD", 81000.0, 2, None, "binance")
        q = self.hub.last["BTC/USD"]
        self.assertEqual(q["price"], 81000.0)
        self.assertAlmostEqual(q["change_pct"], 2.532, places=3)
        self.assertEqual(q["class"], "crypto")

    def test_stale_tick_does_not_overwrite_fresher(self):
        self.hub.on_tick("AAPL", 101.0, 2000, None, "finnhub")
        self.hub.on_tick("AAPL", 100.0, 1000, 99.0, "yahoo")
        self.assertEqual(self.hub.last["AAPL"]["price"], 101.0)
        # but the reference price from the stale message is still learned
        self.hub.on_tick("AAPL", 102.0, 3000, None, "finnhub")
        self.assertIsNotNone(self.hub.last["AAPL"]["change_pct"])

    def test_flush_coalesces_and_filters(self):
        async def go():
            all_q = self.hub.subscribe()
            eur_q = self.hub.subscribe({"EUR/USD"})
            for i in range(100):
                self.hub.on_tick("BTC/USD", 80000.0 + i, i + 1, None, "binance")
            self.hub.on_tick("EUR/USD", 1.12, 1, None, "yahoo")
            self.assertEqual(self.hub.flush(), 2)
            batch = all_q.get_nowait()
            self.assertEqual({t["symbol"] for t in batch}, {"BTC/USD", "EUR/USD"})
            self.assertEqual([t["price"] for t in batch if t["symbol"] == "BTC/USD"], [80099.0])
            self.assertEqual([t["symbol"] for t in eur_q.get_nowait()], ["EUR/USD"])
            self.assertEqual(self.hub.flush(), 0)
        asyncio.run(go())

    def test_slow_subscriber_keeps_newest(self):
        async def go():
            q = self.hub.subscribe()
            for i in range(60):
                self.hub.on_tick("BTC/USD", float(i), i + 1, None, "binance")
                self.hub.flush()
            self.assertEqual(q.qsize(), 50)
            last = None
            while not q.empty():
                last = q.get_nowait()
            self.assertEqual(last[0]["price"], 59.0)
        asyncio.run(go())

    def test_minute_bars(self):
        m = 60_000
        for ts, p in [(0, 10.0), (10_000, 12.0), (20_000, 9.0), (m + 1, 11.0)]:
            self.hub.on_tick("ETH/USD", p, ts, None, "binance")
        bars = self.hub.get_bars("ETH/USD")
        self.assertEqual(bars[0], {"t": 0, "o": 10.0, "h": 12.0, "l": 9.0, "c": 9.0, "n": 3})
        self.assertEqual(bars[1]["t"], m)
        self.assertEqual(self.hub._finished_bars, [("ETH/USD", 0, 10.0, 12.0, 9.0, 9.0, 3)])


class TestBarStorage(unittest.TestCase):

    def test_insert_and_read_bars(self):
        d = tempfile.mkdtemp()
        try:
            db = DatabaseManager(os.path.join(d, "t.db"))
            db.insert_bars([("BTC/USD", 0, 1, 2, 0.5, 1.5, 3), ("BTC/USD", 60000, 1.5, 2, 1, 2, 4)])
            db.insert_bars([("BTC/USD", 60000, 1.5, 3, 1, 2.5, 5)])  # upsert
            bars = db.get_bars("BTC/USD")
            self.assertEqual([b["t"] for b in bars], [0, 60000])
            self.assertEqual(bars[1]["c"], 2.5)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestPriceAPI(unittest.TestCase):
    """Uses TestClient without the context manager, so lifespan (network) doesn't run."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from forex_app.web import app as app_module
        self.hub = app_module.price_hub
        self.hub.last.clear()
        self.hub.on_tick("EUR/USD", 1.1233, 1, 1.1217, "yahoo")
        self.client = TestClient(app_module.app)

    def test_quote_endpoints(self):
        r = self.client.get("/api/quote", params={"symbol": "eurusd"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["symbol"], "EUR/USD")
        self.assertEqual(self.client.get("/api/quote", params={"symbol": "zzz"}).status_code, 404)
        self.assertEqual(self.client.get("/api/quote", params={"symbol": "GBPUSD"}).status_code, 503)
        quotes = self.client.get("/api/quotes", params={"symbols": "EURUSD,BTC"}).json()["quotes"]
        self.assertEqual([q["symbol"] for q in quotes], ["EUR/USD"])

    def test_ws_snapshot_and_resubscribe(self):
        with self.client.websocket_connect("/ws/prices?symbols=EURUSD") as ws:
            msg = ws.receive_json()
            self.assertEqual(msg["type"], "snapshot")
            self.assertEqual([q["symbol"] for q in msg["quotes"]], ["EUR/USD"])
            ws.send_json({"subscribe": ["BTC"]})
            self.assertEqual(ws.receive_json(), {"type": "snapshot", "quotes": []})


if __name__ == "__main__":
    unittest.main()
