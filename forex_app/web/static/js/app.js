// Forex News Dashboard & BM25 Search Application

const state = {
  pair: "ALL",
  sentiment: "ALL",
  impact: "ALL",
  query: "",
  sortBy: "published_at",
  limit: 25,
  offset: 0,
  autoRefreshSeconds: 120,
  countdown: 120,
};

let debounceTimer = null;
let eventSource = null;

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initSearchInput();
  initFilterControls();
  initTickerSim();
  loadPairTabs();
  loadStats();
  loadNews();
  initSSE();
  startCountdownTimer();
});

// ---------------- API Calls ----------------

async function loadNews() {
  const feedList = document.getElementById("articlesList");
  const feedCount = document.getElementById("feedCount");
  const searchMeta = document.getElementById("searchMeta");

  feedList.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-muted);">
    <div class="spin" style="display:inline-block; font-size: 1.5rem; margin-bottom: 8px;">⏳</div>
    <div>Searching & ranking articles...</div>
  </div>`;

  try {
    let url = "";
    const isSearch = state.query.trim().length > 0;

    if (isSearch) {
      const params = new URLSearchParams({
        q: state.query.trim(),
        pair: state.pair,
        sentiment: state.sentiment,
        impact: state.impact,
        limit: state.limit,
        offset: state.offset,
      });
      url = `/api/search?${params.toString()}`;
    } else {
      const params = new URLSearchParams({
        pair: state.pair,
        sentiment: state.sentiment,
        impact: state.impact,
        sort_by: state.sortBy,
        limit: state.limit,
        offset: state.offset,
      });
      url = `/api/news?${params.toString()}`;
    }

    const res = await fetch(url);
    const data = await res.json();

    if (isSearch) {
      feedCount.textContent = `Found ${data.total_hits} results`;
      searchMeta.textContent = `BM25 ranked in ${data.elapsed_ms}ms`;
      renderArticles(data.results, true);
    } else {
      feedCount.textContent = `Showing ${data.articles.length} articles (${data.cached ? "⚡ Memory/Redis Cached" : "SQLite DB"})`;
      searchMeta.textContent = `${data.elapsed_ms}ms`;
      renderArticles(data.articles, false);
    }
  } catch (err) {
    console.error("Error loading news:", err);
    feedList.innerHTML = `<div style="text-align:center; padding: 30px; color: var(--bearish-red);">
      Failed to load articles. Please check server logs.
    </div>`;
  }
}

function renderArticles(articles, isSearch = false) {
  const container = document.getElementById("articlesList");
  if (!articles || articles.length === 0) {
    container.innerHTML = `
      <div style="text-align:center; padding: 50px 20px; background-color: var(--bg-secondary); border-radius: var(--radius-md); border: 1px dashed var(--border-color);">
        <div style="font-size: 2rem; margin-bottom: 8px;">🔍</div>
        <h3 style="font-weight: 700; margin-bottom: 6px;">No articles found</h3>
        <p style="color: var(--text-muted); font-size: 0.9rem;">
          Try refining your search terms, changing the currency pair filter, or clicking "Refresh Live News".
        </p>
      </div>`;
    return;
  }

  container.innerHTML = articles.map(art => {
    const pubDate = formatTimeAgo(art.published_at);
    const sentiment = art.sentiment || "NEUTRAL";
    const impact = art.impact || "MEDIUM";
    const primaryPair = art.primary_pair || "GENERAL";
    const pairs = Array.isArray(art.currency_pairs) ? art.currency_pairs : [primaryPair];
    const tags = Array.isArray(art.tags) ? art.tags : [];

    const sentimentClass = sentiment === "BULLISH" ? "badge-bullish" : sentiment === "BEARISH" ? "badge-bearish" : "badge-neutral";
    const impactClass = impact === "HIGH" ? "badge-high-impact" : impact === "MEDIUM" ? "badge-med-impact" : "badge-low-impact";

    const bm25Badge = (isSearch && art.relevance_score !== undefined)
      ? `<span class="badge badge-bm25" title="BM25 Relevance Score">BM25: ${art.relevance_score}</span>`
      : "";

    const titleHtml = art.title;
    const summaryHtml = art.highlighted_snippet || art.summary || art.content || "";

    const pairBadges = pairs.slice(0, 3).map(p => `
      <span class="badge badge-pair" onclick="selectPair('${p}')">${p}</span>
    `).join("");

    const tagChips = tags.slice(0, 3).map(t => `<span class="tag-chip">${t}</span>`).join("");

    return `
      <div class="article-card">
        <div class="article-header">
          <div style="display:flex; align-items:center; gap: 8px;">
            <span class="article-source">${escapeHtml(art.source)}</span>
            <span style="color: var(--border-light)">•</span>
            <span class="article-time">${pubDate}</span>
          </div>
          <div class="badges-wrap">
            ${pairBadges}
            <span class="badge ${impactClass}">${impact} IMPACT</span>
            <span class="badge ${sentimentClass}">${sentiment}</span>
            ${bm25Badge}
          </div>
        </div>

        <a href="${escapeHtml(art.url)}" target="_blank" rel="noopener noreferrer" class="article-title">
          ${titleHtml}
        </a>

        <div class="article-summary">
          ${summaryHtml}
        </div>

        <div class="article-footer">
          <div class="article-tags">
            ${tagChips}
          </div>
          <a href="${escapeHtml(art.url)}" target="_blank" rel="noopener noreferrer" class="article-link">
            Full Story ↗
          </a>
        </div>
      </div>
    `;
  }).join("");
}

async function loadPairTabs() {
  const container = document.getElementById("pairTabs");
  try {
    const res = await fetch("/api/pairs");
    const data = await res.json();
    const pairs = data.pairs || [];

    let totalCount = 0;
    pairs.forEach(p => totalCount += p.count);

    let html = `
      <button class="pair-tab ${state.pair === 'ALL' ? 'active' : ''}" onclick="selectPair('ALL')">
        ALL <span class="count">${totalCount}</span>
      </button>
    `;

    pairs.forEach(p => {
      html += `
        <button class="pair-tab ${state.pair === p.pair ? 'active' : ''}" onclick="selectPair('${p.pair}')">
          ${p.pair} <span class="count">${p.count}</span>
        </button>
      `;
    });

    container.innerHTML = html;

    // Also populate sidebar pairs list
    renderSidebarPairs(pairs);
  } catch (err) {
    console.error("Error loading pairs:", err);
  }
}

function renderSidebarPairs(pairs) {
  const listEl = document.getElementById("sidebarPairList");
  if (!listEl) return;
  listEl.innerHTML = pairs.slice(0, 8).map(p => `
    <div class="pair-stat-item" onclick="selectPair('${p.pair}')">
      <span class="pair-stat-name">${p.pair}</span>
      <span class="pair-stat-count">${p.count} news</span>
    </div>
  `).join("");
}

async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();

    // Cache Engine Badge
    const cacheBadge = document.getElementById("headerCacheBadge");
    const cacheTitle = document.getElementById("cacheStatusTitle");
    const cacheDetails = document.getElementById("cacheStatusDetails");
    const cacheDot = document.getElementById("cacheStatusDot");

    if (data.cache.is_redis) {
      cacheBadge.innerHTML = `<span class="status-dot redis"></span> Redis Server (6379)`;
      cacheTitle.textContent = "Redis Server (Port 6379)";
      cacheDot.className = "status-dot redis";
      cacheDetails.textContent = `Connected to Redis. ${data.cache.keys_count} keys active.`;
    } else {
      cacheBadge.innerHTML = `<span class="status-dot"></span> Memory Engine (Redis-Ready)`;
      cacheTitle.textContent = "In-Memory + SQLite (Redis-Compatible)";
      cacheDot.className = "status-dot";
      cacheDetails.textContent = `Zero-setup cache active with TTL refresh. ${data.cache.keys_count} keys cached.`;
    }

    // Sentiment Meter
    const sentBreakdown = data.database.sentiment_breakdown || {};
    const bulls = sentBreakdown["BULLISH"] || 0;
    const bears = sentBreakdown["BEARISH"] || 0;
    const neutrals = sentBreakdown["NEUTRAL"] || 0;
    const totalSent = bulls + bears + neutrals || 1;

    const bullPct = Math.round((bulls / totalSent) * 100);
    const bearPct = Math.round((bears / totalSent) * 100);

    const bullFill = document.getElementById("sentimentBullFill");
    const bearFill = document.getElementById("sentimentBearFill");
    const bullPctText = document.getElementById("sentimentBullText");
    const bearPctText = document.getElementById("sentimentBearText");

    if (bullFill) bullFill.style.width = `${bullPct}%`;
    if (bearFill) bearFill.style.width = `${bearPct}%`;
    if (bullPctText) bullPctText.textContent = `${bullPct}% Bullish`;
    if (bearPctText) bearPctText.textContent = `${bearPct}% Bearish`;

    // Total documents indexed in BM25
    const indexCountEl = document.getElementById("indexedDocCount");
    if (indexCountEl) indexCountEl.textContent = data.engine_docs_indexed;

    state.autoRefreshSeconds = data.refresh_interval_seconds || 120;
  } catch (err) {
    console.error("Error loading stats:", err);
  }
}

// ---------------- Interactions ----------------

function selectPair(pairName) {
  state.pair = pairName;
  state.offset = 0;
  loadPairTabs();
  loadNews();
}

function setQuickSearch(query) {
  const input = document.getElementById("searchInput");
  input.value = query;
  state.query = query;
  state.offset = 0;
  document.getElementById("searchClear").style.display = query ? "block" : "none";
  loadNews();
}

function initSearchInput() {
  const input = document.getElementById("searchInput");
  const clearBtn = document.getElementById("searchClear");

  input.addEventListener("input", (e) => {
    const val = e.target.value;
    state.query = val;
    state.offset = 0;
    clearBtn.style.display = val ? "block" : "none";

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      loadNews();
    }, 250);
  });

  clearBtn.addEventListener("click", () => {
    input.value = "";
    state.query = "";
    clearBtn.style.display = "none";
    loadNews();
  });

  // Global keyboard shortcut '/' to focus search
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== input) {
      e.preventDefault();
      input.focus();
      input.select();
    }
  });
}

function initFilterControls() {
  const sentimentSelect = document.getElementById("sentimentFilter");
  const impactSelect = document.getElementById("impactFilter");
  const sortSelect = document.getElementById("sortFilter");

  sentimentSelect.addEventListener("change", (e) => {
    state.sentiment = e.target.value;
    state.offset = 0;
    loadNews();
  });

  impactSelect.addEventListener("change", (e) => {
    state.impact = e.target.value;
    state.offset = 0;
    loadNews();
  });

  sortSelect.addEventListener("change", (e) => {
    state.sortBy = e.target.value;
    state.offset = 0;
    loadNews();
  });
}

async function triggerLiveRefresh() {
  const btn = document.getElementById("refreshBtn");
  const icon = document.getElementById("refreshIcon");
  const origText = btn.innerHTML;

  btn.disabled = true;
  icon.classList.add("spin");

  showToast("Fetching live news from all Forex feeds & central banks...");

  try {
    const res = await fetch("/api/refresh", { method: "POST" });
    const data = await res.json();
    if (data.status === "success") {
      const inserted = data.stats.new_inserted || 0;
      const total = data.stats.total_fetched || 0;
      showToast(`Refresh complete! Fetched ${total} items, ${inserted} new articles indexed.`);
      state.countdown = state.autoRefreshSeconds;
      await loadStats();
      await loadPairTabs();
      await loadNews();
    } else {
      showToast("Refresh encountered an issue. See logs.");
    }
  } catch (err) {
    showToast("Error during refresh: " + err.message);
  } finally {
    btn.disabled = false;
    icon.classList.remove("spin");
  }
}

// ---------------- Countdown Timer ----------------

function startCountdownTimer() {
  const timerEl = document.getElementById("refreshTimer");
  setInterval(() => {
    state.countdown--;
    if (state.countdown <= 0) {
      state.countdown = state.autoRefreshSeconds;
      triggerLiveRefresh();
    }
    if (timerEl) {
      timerEl.textContent = `${state.countdown}s`;
    }
  }, 1000);
}

// ---------------- Server-Sent Events (SSE) ----------------

function initSSE() {
  if (!window.EventSource) return;
  try {
    eventSource = new EventSource("/api/events");
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "new_articles") {
          showToast(`⚡ ${data.count} new Forex news articles ingested & indexed!`);
          loadPairTabs();
          loadStats();
          loadNews();
        }
      } catch (e) {}
    };
  } catch (e) {
    console.warn("SSE connection error:", e);
  }
}

// ---------------- Modals ----------------

function openRedisModal() {
  document.getElementById("redisModal").classList.add("active");
}

function closeRedisModal() {
  document.getElementById("redisModal").classList.remove("active");
}

async function testRedisConnection() {
  const urlInput = document.getElementById("redisTestUrl");
  const resultEl = document.getElementById("redisTestResult");
  const url = urlInput.value.trim() || "redis://localhost:6379/0";

  resultEl.innerHTML = `<span style="color: var(--accent-cyan)">Testing connection to ${url}...</span>`;

  try {
    const res = await fetch("/api/redis/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ redis_url: url }),
    });
    const data = await res.json();
    if (data.success) {
      resultEl.innerHTML = `<span style="color: var(--bullish-green)">✔ ${data.message}</span>`;
      loadStats();
    } else {
      resultEl.innerHTML = `<span style="color: var(--bearish-red)">✖ ${data.message}</span>`;
    }
  } catch (err) {
    resultEl.innerHTML = `<span style="color: var(--bearish-red)">Error: ${err.message}</span>`;
  }
}

function copyCode(btn, text) {
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => btn.textContent = orig, 1500);
  });
}

// ---------------- Helpers ----------------

function showToast(msg) {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function formatTimeAgo(isoString) {
  if (!isoString) return "just now";
  try {
    const date = new Date(isoString);
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return "just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  } catch {
    return isoString;
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function initTickerSim() {
  // Minor live jitter to ticker prices every 4 seconds to reflect live market heartbeat
  const tickerItems = document.querySelectorAll(".ticker-item");
  setInterval(() => {
    tickerItems.forEach(item => {
      const valEl = item.querySelector(".val");
      if (!valEl) return;
      let cur = parseFloat(valEl.textContent.replace("$", "").replace(",", ""));
      if (isNaN(cur)) return;
      let delta = (Math.random() - 0.49) * 0.0008 * cur;
      let newVal = cur + delta;
      if (cur > 100) {
        valEl.textContent = newVal.toFixed(2);
      } else {
        valEl.textContent = newVal.toFixed(4);
      }
    });
  }, 3500);
}
