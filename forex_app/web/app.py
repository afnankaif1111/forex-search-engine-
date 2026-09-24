"""FastAPI Web Application and REST API for Forex News & Search Engine."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import ALL_PAIRS, MAJOR_PAIRS, RSS_FEEDS
from ..engine.search import search_engine
from ..fetcher.news_service import news_service
from ..storage.cache import cache
from ..storage.database import db

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB, seed if empty, build BM25 search index, start background worker
    logger.info("Initializing Forex News Application...")
    news_service.initialize()
    news_service.start_background_worker()
    yield
    # Shutdown
    news_service.stop_background_worker()
    logger.info("Forex News Application shutdown complete.")


app = FastAPI(
    title="Forex News Search & Redis Caching Engine",
    description="Live Forex news aggregator, Redis/In-Memory cache, and BM25 search engine.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------- Web Pages ----------------

@app.get("/", response_class=HTMLResponse)
async def home_page():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Index file not found</h1>", status_code=500)
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/reels", response_class=HTMLResponse)
async def reels_page():
    reels_file = TEMPLATES_DIR / "reels.html"
    if not reels_file.exists():
        return HTMLResponse("<h1>Reels file not found</h1>", status_code=500)
    with open(reels_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ---------------- REST APIs ----------------

@app.get("/api/reels")
async def get_reels(
    genre: str = Query("all", description="Genre or category filter"),
    limit: int = Query(40, ge=5, le=100),
):
    """Retrieve reels filtered by genre."""
    genre_clean = genre.strip()
    if genre_clean.lower() == "all":
        articles = db.get_articles(limit=limit, sort_by="published_at")
    elif genre_clean.upper() in [p.upper() for p in ALL_PAIRS]:
        articles = db.get_articles(limit=limit, pair=genre_clean, sort_by="published_at")
    elif genre_clean.lower() == "high_impact":
        articles = db.get_articles(limit=limit, impact="HIGH", sort_by="published_at")
    elif genre_clean.lower() == "bullish":
        articles = db.get_articles(limit=limit, sentiment="BULLISH", sort_by="published_at")
    elif genre_clean.lower() == "bearish":
        articles = db.get_articles(limit=limit, sentiment="BEARISH", sort_by="published_at")
    elif genre_clean.lower() == "central_banks":
        articles, _ = search_engine.search(query="central bank fed ecb rate powell lagarde", limit=limit)
    elif genre_clean.lower() == "inflation":
        articles, _ = search_engine.search(query="inflation cpi ppi price cost", limit=limit)
    elif genre_clean.lower() == "gold":
        articles = db.get_articles(limit=limit, pair="XAU/USD", sort_by="published_at")
    elif genre_clean.lower() == "technicals":
        articles, _ = search_engine.search(query="technical support resistance breakout chart", limit=limit)
    else:
        articles, _ = search_engine.search(query=genre_clean, limit=limit)

    return {
        "genre": genre,
        "count": len(articles),
        "reels": articles,
    }

@app.get("/api/news")
async def get_news(
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    pair: Optional[str] = Query("ALL"),
    sentiment: Optional[str] = Query("ALL"),
    impact: Optional[str] = Query("ALL"),
    sort_by: str = Query("published_at"),
):
    """Retrieve filtered articles. Uses in-memory/Redis cache for standard requests."""
    t0 = time.perf_counter()

    # Fast cache check for standard default view (first page, all pairs, no filters)
    is_default_view = (
        pair == "ALL"
        and sentiment == "ALL"
        and impact == "ALL"
        and offset == 0
        and sort_by == "published_at"
    )
    if is_default_view:
        cached_articles = cache.get_json("forex:latest_articles")
        if cached_articles and len(cached_articles) >= limit:
            return {
                "articles": cached_articles[:limit],
                "total": len(cached_articles),
                "cached": True,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

    # Query from database
    articles = db.get_articles(
        limit=limit,
        offset=offset,
        pair=pair,
        sentiment=sentiment,
        impact=impact,
        sort_by=sort_by,
    )
    stats = db.get_stats()

    return {
        "articles": articles,
        "total": stats["total_articles"],
        "cached": False,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/api/search")
async def search_articles(
    q: str = Query(..., min_length=1, description="Search query string"),
    pair: Optional[str] = Query("ALL"),
    sentiment: Optional[str] = Query("ALL"),
    impact: Optional[str] = Query("ALL"),
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """BM25 Ranked Search across the complete Forex news database with keyword highlighting."""
    t0 = time.perf_counter()
    results, total_hits = search_engine.search(
        query=q,
        pair=pair,
        sentiment=sentiment,
        impact=impact,
        limit=limit,
        offset=offset,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "query": q,
        "results": results,
        "total_hits": total_hits,
        "limit": limit,
        "offset": offset,
        "elapsed_ms": elapsed_ms,
        "algorithm": "Okapi BM25 (Title 2.5x, Pairs 3.0x, Summary 1.2x)",
    }


@app.post("/api/refresh")
async def trigger_refresh():
    """Trigger an immediate live fetch from all Forex news APIs and RSS feeds."""
    t0 = time.perf_counter()
    try:
        # Run fetch in threadpool to avoid blocking event loop
        stats = await asyncio.to_thread(news_service.fetch_all_sources)
        elapsed_s = round(time.perf_counter() - t0, 2)
        return {
            "status": "success",
            "elapsed_seconds": elapsed_s,
            "stats": stats,
        }
    except Exception as e:
        logger.error("Manual refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Refresh failed: {e}")


@app.get("/api/pairs")
async def get_pairs():
    """List of all supported currency pairs and current article counts."""
    counts = db.get_pair_counts()
    pair_list = []
    for p in ALL_PAIRS:
        pair_list.append({
            "pair": p,
            "count": counts.get(p, 0),
            "is_major": p in MAJOR_PAIRS,
        })
    return {
        "pairs": pair_list,
        "total_tracked": len(ALL_PAIRS),
    }


@app.get("/api/stats")
async def get_stats():
    """Comprehensive system and market statistics."""
    db_stats = db.get_stats()
    cache_status = cache.get_status()
    return {
        "database": db_stats,
        "cache": cache_status,
        "engine_docs_indexed": search_engine.n_docs,
        "feeds_configured": len(RSS_FEEDS),
        "refresh_interval_seconds": news_service.refresh_interval,
    }


@app.get("/api/redis/info")
async def get_redis_info():
    """Detailed Redis diagnostics and setup advice."""
    return cache.get_status()


class RedisTestPayload(BaseModel):
    redis_url: str = "redis://localhost:6379/0"


@app.post("/api/redis/test")
async def test_redis_connection(payload: RedisTestPayload):
    """Test connection to a specified Redis server URL."""
    try:
        import redis
        client = redis.Redis.from_url(
            payload.redis_url,
            decode_responses=True,
            socket_timeout=1.5,
            socket_connect_timeout=1.5,
        )
        client.ping()
        return {
            "success": True,
            "message": f"Successfully connected and pinged Redis at {payload.redis_url}!",
        }
    except ImportError:
        return {
            "success": False,
            "message": "Python 'redis' package is not installed. Run 'pip install redis' to use external Redis.",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to connect to Redis at {payload.redis_url}: {e}",
        }


class SettingsPayload(BaseModel):
    refresh_interval: Optional[int] = None


@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    """Update runtime configuration like refresh interval."""
    if payload.refresh_interval is not None:
        if payload.refresh_interval < 10:
            raise HTTPException(status_code=400, detail="Refresh interval must be at least 10 seconds.")
        news_service.refresh_interval = payload.refresh_interval
    return {
        "status": "updated",
        "refresh_interval": news_service.refresh_interval,
    }


# ---------------- Server-Sent Events (SSE) ----------------

@app.get("/api/events")
async def sse_events(request: Request):
    """Stream live news events to connected browser tabs via SSE."""
    queue = news_service.subscribe()

    async def event_generator():
        try:
            # Send initial connection heartbeat
            yield f"data: {json.dumps({'type': 'connected', 'timestamp': time.time()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # Wait up to 25 seconds for an event, then send keepalive comment
                    message = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            news_service.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
