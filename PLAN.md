# Market Intelligence App — Build Plan

Goal: one web app that (1) streams live prices for **forex, stocks and crypto**
as fast as the sources allow, (2) pulls relevant news/web content for any
keyword, and (3) ranks it with **hybrid search** (BM25 + vector/semantic).

---

## 1. What already exists (analysis)

The repo is two projects in one:

| Part | What it is | State |
|---|---|---|
| `searchengine/` | A serious, benchmarked search engine: C MaxScore BM25 kernel, compressed index, spell/did-you-mean (`spell.py`), autocomplete (`suggest.py`), phrase search, live incremental index (`live/`), crawler + PageRank, sharding, and **dense search**: ONNX bi-encoder (`encoder.py`), exact/PQ vector index (`dense.py`), IVF ANN (`ann.py`), hybrid fusion (`search_hybrid.py`), cross-encoder rerank (`cross_encoder.py`). | Works; built for MS MARCO (8.8M passages). Dense parts need `onnxruntime` + model files in `models/` — **neither is present on this machine**. |
| `forex_app/` (FXPulse) | FastAPI app: 6 RSS feeds → regex NLP (pair detection, keyword sentiment, impact) → SQLite → pure-Python BM25 with field boosts → in-memory/Redis cache → SSE push → terminal UI + "reels" UI. | Runs; 8/8 tests pass; DB holds 269 articles. **News only — no price data at all.** Does not use `searchengine/` at all. |

### Gaps / problems to fix first
1. **Slow ingestion.** `news_service.fetch_all_sources` fetches feeds *one after another* with a 6s timeout each (worst case ~36s per refresh). No ETag / If-Modified-Since, so every refresh re-downloads everything.
2. **No market prices** — forex, stocks, crypto quotes are missing entirely.
3. **No keyword-driven web search.** Search only covers what the 6 fixed feeds happened to return.
4. **Keyword-only relevance.** "dovish Fed" won't match "Powell signals rate cuts". This is what vector search fixes.
5. `_process_and_store` re-queries "latest row" to get the new id — racy under concurrency; use `cursor.lastrowid`.
6. Sentiment is a keyword list — weak on finance language ("beats estimates but guidance cut").
7. Asset universe is hard-coded to 18 pairs; no stock tickers, only BTC for crypto.
8. `allow_origins=["*"]` + `allow_credentials=True` in CORS — tighten before deploying.

---

## 2. Target architecture

```
                ┌───────────── PRICE PLANE (push, ms latency) ─────────────┐
 Binance/Coinbase WS ─┐                                                    │
 Finnhub WS (stocks,FX)┼─► price_hub (asyncio) ─► in-mem last-tick + Redis ─┼─► WebSocket /ws/prices ─► UI tickers/charts
 OANDA practice stream ┘        │                                          │
                                └─► 1-min OHLC bars (SQLite/DuckDB)         │
                └──────────────────────────────────────────────────────────┘

                ┌───────────── NEWS PLANE (pull, seconds latency) ─────────┐
 RSS feeds (parallel, ETag) ─┐                                             │
 GDELT DOC API (keyword)     ┼─► normalize ─► dedup ─► NLP tags ─► SQLite  │
 Google News RSS ?q=keyword  │                    │                        │
 SEC EDGAR / crypto RSS     ─┘                    ├─► BM25 index (live)    │
                                                  └─► embeddings (vectors) │
                └──────────────────────────────────────────────────────────┘

 GET /api/search?q=  ─►  query understanding (asset detect, filters)
                     ─►  BM25 top-100  ∪  vector top-100
                     ─►  fusion (RRF / α-blend) × recency decay
                     ─►  optional cross-encoder rerank top-20
                     ─►  results + live price of detected asset
```

Key design rule: **prices are pushed over WebSockets, never polled**; news is
pulled concurrently and cached. That is where "as fast as possible" comes from.

### Data sources (free or free-tier)
| Asset | Real-time source | Notes |
|---|---|---|
| Crypto | Binance public WS (`wss://stream.binance.com:9443/ws/<sym>@trade`), Coinbase WS | No key, true real-time |
| Stocks + FX | Finnhub WS (`wss://ws.finnhub.io?token=`) | Free key; `config.py` already reads `FINNHUB_API_KEY` |
| Forex | OANDA practice-account streaming API | Free demo account, real streaming FX |
| Fallback / history | Alpha Vantage, Twelve Data, Frankfurter (ECB daily) | Rate-limited; use for backfill only |
| Keyword news | GDELT DOC 2.0 API, Google News RSS search, existing RSS list | No key |
| Filings | SEC EDGAR RSS / full-text search | No key, needs a User-Agent with contact email |

Check each provider's current terms and rate limits before relying on them.

---

## 3. Executable workflow

Each phase ends with a **check** command. Don't start the next phase until it passes.

### Phase 0 — Environment (½ day)
```bash
cd ~/Downloads/search-engine-main
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn httpx websockets numpy onnxruntime tokenizers redis feedparser python-dotenv pytest
cp .env.example .env   # FINNHUB_API_KEY=..., OANDA_TOKEN=... (create this file)
```
Check: `python3 -m pytest tests/test_forex_app.py -q` → all pass.

### Phase 1 — Fast news ingestion (1 day)
- New `forex_app/fetcher/async_fetch.py`: `httpx.AsyncClient(http2=True)` + `asyncio.gather` over all feeds, 4s timeout, store `ETag`/`Last-Modified` per feed and send them back (304 = skip).
- Fix `_process_and_store` to use `cursor.lastrowid`.
- Near-duplicate guard: normalized-title hash (later replaced by vector similarity, Phase 4).
- Add crypto + stock RSS (CoinDesk, Cointelegraph, SEC EDGAR, Nasdaq).

Check: `curl -XPOST localhost:8000/api/refresh` → `elapsed_seconds` < 5 and stays < 1 on a no-change refresh.

### Phase 2 — Live price plane (2 days)
- `forex_app/prices/hub.py`: one asyncio task per provider, auto-reconnect with backoff, normalize to `{symbol, asset_class, bid, ask, last, ts, source}`.
- Keep last tick in a dict + Redis hash `px:<symbol>`; roll 1-min OHLC bars into a `bars` table.
- `forex_app/web/app.py`: add `WebSocket /ws/prices` (client subscribes to symbols) and `GET /api/quote/{symbol}`.
- Expand config: `ASSETS = {"forex": [...], "stocks": ["AAPL","NVDA",...], "crypto": ["BTCUSDT","ETHUSDT","SOLUSDT",...]}`.

Check: `python3 -m forex_app.prices.hub --symbols BTCUSDT,EUR/USD,AAPL --seconds 10` prints ticks; tick-to-browser latency < 200ms.

### Phase 3 — Keyword web search (1 day)
- `forex_app/fetcher/keyword_search.py`: on a query, fire GDELT + Google News RSS + local index **in parallel**; return local hits immediately and stream web hits as they arrive (SSE or a second request).
- Every fetched web result is ingested into the DB + indexes, so the corpus grows with what users search.
- Cache per-query results 60–120s in Redis.

Check: `curl 'localhost:8000/api/search?q=nvidia+earnings&web=1'` returns local hits in < 50ms and web hits in < 3s.

### Phase 4 — Vector / semantic search (2–3 days)
Reuse the repo's own engine — it already solved this:
1. **Model.** Export a small embedding model to ONNX into `models/bge-small` or `models/e5-small` (384-dim, fast on CPU). `encoder.py` already has the pooling/prefix configs for BGE/E5/MiniLM. Optionally compare with a finance-tuned model later.
   ```bash
   pip install optimum[exporters]
   optimum-cli export onnx --model BAAI/bge-small-en-v1.5 models/bge
   ```
2. **Embed on insert.** In `_process_and_store`, embed `title + summary` and append to a vector store. At this corpus size (thousands → low millions) use `dense.py`'s **exact f32** mode (the repo's own notes: PQ only pays off above ~1M docs). Alternative if you prefer SQL: `sqlite-vec` extension in the same DB.
3. **Hybrid query** (`forex_app/engine/hybrid.py`):
   - BM25 top-100 (existing engine) ∪ vector top-100 (matrix @ query_vec).
   - Fuse with **Reciprocal Rank Fusion** (robust, no tuning) or the repo's min-max α-blend starting at `ALPHA_UNKNOWN_DOMAIN = 0.5` (the repo measured that a wrong α is *worse than BM25*, so don't guess 0.1).
   - Multiply by recency decay `exp(-age_hours / 48)` — news relevance rots fast.
   - Optional: `cross_encoder.py` rerank of top-20 for the best quality.
4. **Endpoint:** `/api/search?q=&mode=bm25|semantic|hybrid` (default hybrid) so you can A/B.
5. **Evaluate**, don't eyeball: write 50 real queries, label top-10 results (the repo's `bench/webeval/` shows the pooled-judgment method), compare nDCG@10 for the three modes.

Check: `python3 -m pytest tests/test_hybrid.py` and nDCG@10 hybrid ≥ BM25 on your 50 queries; query p95 < 100ms.

### Phase 5 — UI (2 days)
- Ticker strip + mini charts (lightweight-charts by TradingView) fed from `/ws/prices`.
- Search box: results grouped by asset, with the detected asset's live quote pinned on top.
- Watchlist panel.

### Phase 6 — Hardening (1 day)
- Tighten CORS, rate-limit `/api/search` and `/api/refresh`, health endpoint, structured logs, Dockerfile, run Redis for real.

**Total: roughly 10–12 working days solo.**

---

## 4. More ideas built on vector search

1. **Story clustering** — articles with cosine > 0.88 within 24h form one "story"; show one card with N sources instead of 6 duplicates. Big quality win for news feeds.
2. **Semantic alerts** — user saves "ECB turning hawkish"; every new article is embedded and compared against saved alert vectors; match → push via SSE/Telegram/email.
3. **Price-reaction labelling** — for every article, record the asset's move 5m / 1h / 1d after publication. Lets you rank by "news that actually moved the market" and later train a model on it.
4. **"Why is X moving?" (RAG)** — when a price jumps > N σ, retrieve the top hybrid hits from the last hour and have an LLM (Claude API) write a 3-line cited explanation.
5. **Query understanding** — detect assets/tickers/central banks in the query and turn them into filters + boosts ("gold fed" → `XAU/USD` filter + "federal reserve" expansion). `classifier.py` already has the dictionaries.
6. **Finance sentiment model** — replace keyword sentiment with FinBERT (ONNX via the same `encoder.py` machinery); per-asset sentiment index chart over time.
7. **Economic calendar** — ingest CPI/NFP/rate-decision schedule; link articles to the event they discuss; countdown on the UI.
8. **"More like this"** — one button: use the article's own vector as the query.
9. **Semantic autocomplete** — combine existing `suggest.py` prefix suggest with nearest past queries in vector space; `spell.py` for did-you-mean on tickers.
10. **Cross-asset correlation view** — rolling correlation of the 1-min bars (EUR/USD vs DXY vs gold vs BTC) next to the news timeline.
11. **Multilingual news** — use a multilingual embedding model (e.g. multilingual-e5) so Japanese/German central-bank news matches English queries.
12. **Source credibility weighting** — `config.py` already has `reliability` per feed; fold it into the ranking score.
