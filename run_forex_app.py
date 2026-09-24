#!/usr/bin/env python3
"""Launcher for Forex News Aggregator & Search Engine Application."""

import logging
import socket
import sys
import uvicorn

from forex_app.config import SERVER_HOST, SERVER_PORT
from forex_app.storage.cache import cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("forex_app")


def find_available_port(start_port: int) -> int:
    """Find an open port starting from start_port."""
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port


def main():
    port = find_available_port(SERVER_PORT)

    print("\n" + "=" * 65)
    print("⚡  FXPULSE — FOREX NEWS SEARCH ENGINE & REDIS CACHE TERMINAL")
    print("=" * 65)
    print(f"🚀  Web Dashboard:        http://localhost:{port}")
    print(f"📚  API Documentation:    http://localhost:{port}/docs")
    print(f"💾  Cache Engine:         {cache.get_status()['engine']}")
    print(f"🔄  Auto-Refresh:         Every 120 seconds in background")
    print(f"🔍  Search Algorithm:     Okapi BM25 Multi-Field Ranked Retrieval")
    print("=" * 65 + "\n")

    uvicorn.run(
        "forex_app.web.app:app",
        host=SERVER_HOST,
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
