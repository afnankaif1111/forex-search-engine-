// FXPulse — markets news search + live prices

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
  marketClass: "forex",
  shown: 0,
};

let debounceTimer = null;
let eventSource = null;

document.addEventListener("DOMContentLoaded", () => {
  initSearchInput();
  initFilterControls();
  initMarketTabs();
  initLivePrices();
  loadPairTabs();
  loadStats();
  loadNews();
  initSSE();
  startCountdownTimer();
  const toTop = document.getElementById("toTop");
  window.addEventListener("scroll", () => toTop.classList.toggle("hide", window.scrollY < 600), { passive: true });
});

// ---------------- News ----------------

async function loadNews(append = false) {
  const feedList = document.getElementById("articlesList");
  const feedCount = document.getElementById("feedCount");
  const searchMeta = document.getElementById("searchMeta");

  if (!append) {
    feedList.innerHTML = Array.from({ length: 4 }, () => `
      <div class="glass flex flex-col gap-2 rounded-2xl p-5">
        <div class="skeleton h-3 w-32 bg-white/5"></div>
        <div class="skeleton h-5 w-3/4 bg-white/5"></div>
        <div class="skeleton h-3 w-full bg-white/5"></div>
      </div>`).join("");
  }
  const loadMoreBtn = document.getElementById("loadMoreBtn");
  loadMoreBtn.disabled = true;

  try {
    const isSearch = state.query.trim().length > 0;
    const params = new URLSearchParams({
      pair: state.pair,
      sentiment: state.sentiment,
      impact: state.impact,
      limit: state.limit,
      offset: state.offset,
    });
    if (isSearch) params.set("q", state.query.trim());
    else params.set("sort_by", state.sortBy);

    const res = await fetch(`/api/${isSearch ? "search" : "news"}?${params}`);
    const data = await res.json();

    const items = isSearch ? data.results : data.articles;
    const total = isSearch ? data.total_hits : data.total;
    if (append) state.shown += items.length; else state.shown = items.length;
    feedCount.textContent = isSearch
      ? `${total} result${total === 1 ? "" : "s"} for “${state.query.trim()}”`
      : `Showing ${state.shown} of ${total} articles`;
    searchMeta.textContent = `${data.elapsed_ms} ms${!isSearch && data.cached ? " · cached" : ""}`;
    renderArticles(items, isSearch, append);
    loadMoreBtn.style.display = state.shown < total ? "" : "none";
    loadMoreBtn.textContent = `Load ${Math.min(state.limit, total - state.shown)} more`;
  } catch (err) {
    console.error("Error loading news:", err);
    feedList.innerHTML = `<div class="glass rounded-2xl py-10 text-center text-sm text-white/50">Couldn't load articles. <button class="link" onclick="loadNews()">Try again</button></div>`;
  } finally {
    loadMoreBtn.disabled = false;
  }
}

function loadMore() {
  state.offset += state.limit;
  loadNews(true);
}

function sentimentClass(s) {
  return s === "BULLISH" ? "up" : s === "BEARISH" ? "down" : "text-white/40";
}

function renderArticles(articles, isSearch = false, append = false) {
  const container = document.getElementById("articlesList");
  if (!articles || articles.length === 0) {
    if (append) return;
    container.innerHTML = `
      <div class="glass rounded-2xl px-6 py-16 text-center">
        <div class="text-3xl">🔍</div>
        <div class="mt-3 font-medium">Nothing matched${state.query ? ` “${escapeHtml(state.query)}”` : ""}</div>
        <div class="mx-auto mt-1 max-w-sm text-sm text-white/40">Try different words, relax the filters, or refresh the feeds.</div>
        <div class="mt-5 flex flex-wrap justify-center gap-2">
          <button class="btn btn-sm rounded-full border-white/15 bg-white/5 font-normal" onclick="resetFilters()">Reset filters</button>
          <button class="btn btn-sm rounded-full border-white/15 bg-white/5 font-normal" onclick="setQuickSearch('Bitcoin')">Try “Bitcoin”</button>
        </div>
      </div>`;
    return;
  }

  const html = articles.map((art, i) => {
    const sentiment = art.sentiment || "NEUTRAL";
    const impact = art.impact || "MEDIUM";
    const pairs = Array.isArray(art.currency_pairs) && art.currency_pairs.length
      ? art.currency_pairs : [art.primary_pair || "GENERAL"];
    const url = escapeHtml(art.url);

    const pairBadges = pairs.filter(p => p !== "GENERAL").slice(0, 3).map(p =>
      `<button class="badge badge-sm badge-ghost border-white/10 font-mono" onclick="selectPair('${escapeHtml(p)}')">${escapeHtml(p)}</button>`
    ).join("");
    const impactBadge = impact === "HIGH"
      ? `<span class="badge badge-sm gap-1 border-orange-400/30 bg-orange-400/15 text-orange-300">⚡ High impact</span>` : "";
    const sentPill = sentiment === "BULLISH"
      ? `<span class="pill-up rounded-full px-2 py-0.5 text-[11px] font-medium">▲ Bullish</span>`
      : sentiment === "BEARISH"
        ? `<span class="pill-down rounded-full px-2 py-0.5 text-[11px] font-medium">▼ Bearish</span>`
        : `<span class="pill-flat rounded-full px-2 py-0.5 text-[11px] font-medium">● Neutral</span>`;
    const cardCls = sentiment === "BULLISH" ? "bull" : sentiment === "BEARISH" ? "bear" : "";
    const initial = escapeHtml((art.source || "?").trim().charAt(0).toUpperCase());
    const score = (isSearch && art.relevance_score !== undefined)
      ? `<span class="font-mono text-white/30" title="BM25 relevance">${art.relevance_score}</span>` : "";

    // title/snippet may contain the engine's <mark> highlight tags
    return `
      <article class="card fx-card glass group rounded-2xl p-5 pl-6 ${cardCls}" data-tilt="4" style="animation-delay:${Math.min(i, 10) * 35}ms">
        <div class="fx-edge"></div>
        <div class="mb-2 flex flex-wrap items-center gap-x-2.5 gap-y-1.5 text-xs text-white/40">
          <span class="grid h-5 w-5 place-items-center rounded-md bg-white/10 text-[10px] font-semibold text-white/70">${initial}</span>
          <span class="text-white/70">${escapeHtml(art.source)}</span>
          <span>·</span>
          <span>${formatTimeAgo(art.published_at)}</span>
          ${sentPill}
          ${score ? `<span class="ml-auto">${score}</span>` : ""}
        </div>
        <a href="${url}" target="_blank" rel="noopener noreferrer" class="block text-[17px] font-medium leading-snug text-white group-hover:text-indigo-200">
          ${art.title}
        </a>
        <p class="mt-1.5 line-clamp-2 text-sm leading-relaxed text-white/50">
          ${art.highlighted_snippet || art.summary || art.content || ""}
        </p>
        ${(pairBadges || impactBadge) ? `<div class="mt-3 flex flex-wrap gap-1.5">${impactBadge}${pairBadges}</div>` : ""}
      </article>`;
  }).join("");
  if (append) container.insertAdjacentHTML("beforeend", html);
  else container.innerHTML = html;
}

function resetFilters() {
  Object.assign(state, { pair: "ALL", sentiment: "ALL", impact: "ALL", query: "", sortBy: "published_at", offset: 0 });
  ["sentimentFilter", "impactFilter", "sortFilter"].forEach((id, i) => {
    document.getElementById(id).value = ["ALL", "ALL", "published_at"][i];
  });
  document.getElementById("searchInput").value = "";
  document.getElementById("searchClear").style.display = "none";
  loadPairTabs();
  loadNews();
}

async function loadPairTabs() {
  const container = document.getElementById("pairTabs");
  try {
    const res = await fetch("/api/pairs");
    const data = await res.json();
    const pairs = (data.pairs || []).filter(p => p.count > 0);

    const tab = (value, label, count) => `
      <button class="btn btn-sm flex-none rounded-full font-normal ${state.pair === value ? "border-white/30 bg-white/15 text-white" : "border-transparent bg-transparent text-white/60"}" onclick="selectPair('${value}')">
        ${label}${count !== null ? `<span class="font-mono text-xs opacity-40">${count}</span>` : ""}
      </button>`;

    container.innerHTML = tab("ALL", "All", null) +
      pairs.map(p => tab(p.pair, `<span class="font-mono">${p.pair}</span>`, p.count)).join("");

    const listEl = document.getElementById("sidebarPairList");
    const top = [...pairs].sort((a, b) => b.count - a.count).slice(0, 6);
    const max = top.length ? top[0].count : 1;
    listEl.innerHTML = top.map(p => `
      <li>
        <button class="group w-full rounded-lg px-2 py-1.5 text-left transition hover:bg-white/5" onclick="selectPair('${p.pair}')">
          <div class="flex items-center justify-between">
            <span class="font-mono text-white/80 group-hover:text-white">${p.pair}</span>
            <span class="font-mono text-xs text-white/40">${p.count}</span>
          </div>
          <div class="mt-1 h-0.5 w-full rounded bg-white/5"><div class="h-0.5 rounded bg-indigo-400/60" style="width:${Math.round(p.count / max * 100)}%"></div></div>
        </button>
      </li>`).join("");
  } catch (err) {
    console.error("Error loading pairs:", err);
  }
}

async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();

    const dot = document.getElementById("cacheStatusDot");
    const label = document.getElementById("cacheStatusLabel");
    const details = document.getElementById("cacheStatusDetails");
    if (data.cache.is_redis) {
      dot.className = "inline-block h-2 w-2 rounded-full bg-green-500";
      label.textContent = "Redis";
      details.textContent = `Connected to Redis · ${data.cache.keys_count} keys`;
    } else {
      dot.className = "inline-block h-2 w-2 rounded-full bg-white/40";
      label.textContent = "Memory cache";
      details.textContent = `Built-in in-memory cache with SQLite storage · ${data.cache.keys_count} keys`;
    }

    const sb = data.database.sentiment_breakdown || {};
    const bulls = sb.BULLISH || 0, bears = sb.BEARISH || 0, neutral = sb.NEUTRAL || 0;
    const total = bulls + bears + neutral || 1;
    const bullPct = Math.round(bulls / total * 100);
    const bearPct = Math.round(bears / total * 100);
    document.getElementById("sentimentBullFill").style.width = `${bullPct}%`;
    document.getElementById("sentimentBearFill").style.width = `${bearPct}%`;
    document.getElementById("sentimentBullText").textContent = `${bullPct}% bullish`;
    document.getElementById("sentimentBearText").textContent = `${bearPct}% bearish`;

    const mood = document.getElementById("moodLabel");
    const diff = bullPct - bearPct;
    mood.textContent = diff > 10 ? "Leaning bullish" : diff < -10 ? "Leaning bearish" : "Mixed / neutral";
    mood.className = "text-lg font-semibold " + (diff > 10 ? "up" : diff < -10 ? "down" : "text-white/80");

    FX.countUp(document.getElementById("indexedDocCount"), data.engine_docs_indexed);
    FX.countUp(document.getElementById("heroCount"), data.engine_docs_indexed);

    if (data.prices) {
      FX.countUp(document.getElementById("liveSymbolCount"), data.prices.symbols_live, 700);
      document.getElementById("liveSourceDesc").textContent =
        data.prices.providers.map(p => `${p.active}${p.connected ? "" : " (down)"}`).join(" · ");
    }

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
  document.getElementById("searchClear").style.display = query ? "" : "none";
  loadNews();
}

function initSearchInput() {
  const input = document.getElementById("searchInput");
  const clearBtn = document.getElementById("searchClear");

  input.addEventListener("input", (e) => {
    state.query = e.target.value;
    state.offset = 0;
    clearBtn.style.display = state.query ? "" : "none";
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(loadNews, 250);
  });

  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    input.value = "";
    state.query = "";
    clearBtn.style.display = "none";
    loadNews();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== input && !(e.target instanceof HTMLInputElement)) {
      e.preventDefault();
      input.focus();
      input.select();
    }
  });
}

function initFilterControls() {
  const bind = (id, key) => document.getElementById(id).addEventListener("change", (e) => {
    state[key] = e.target.value;
    state.offset = 0;
    loadNews();
  });
  bind("sentimentFilter", "sentiment");
  bind("impactFilter", "impact");
  bind("sortFilter", "sortBy");
}

async function triggerLiveRefresh() {
  const btn = document.getElementById("refreshBtn");
  const icon = document.getElementById("refreshIcon");
  btn.disabled = true;
  icon.classList.add("spin");

  try {
    const res = await fetch("/api/refresh", { method: "POST" });
    const data = await res.json();
    if (data.status === "success") {
      const n = data.stats.new_inserted || 0;
      showToast(n ? `${n} new articles` : "Up to date");
      state.countdown = state.autoRefreshSeconds;
      await Promise.all([loadStats(), loadPairTabs(), loadNews()]);
    } else {
      showToast("Refresh failed — see server logs");
    }
  } catch (err) {
    showToast("Refresh failed: " + err.message);
  } finally {
    btn.disabled = false;
    icon.classList.remove("spin");
  }
}

function startCountdownTimer() {
  const timerEl = document.getElementById("refreshTimer");
  setInterval(() => {
    state.countdown--;
    if (state.countdown <= 0) {
      state.countdown = state.autoRefreshSeconds;
      triggerLiveRefresh();
    }
    timerEl.textContent = `${state.countdown}s`;
  }, 1000);
}

function initSSE() {
  if (!window.EventSource) return;
  eventSource = new EventSource("/api/events");
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "new_articles") {
        showToast(`${data.count} new articles`);
        loadPairTabs();
        loadStats();
        loadNews();
      }
    } catch (e) {}
  };
}

// ---------------- Cache modal ----------------

function openRedisModal() {
  document.getElementById("redisModal").showModal();
}

function closeRedisModal() {
  document.getElementById("redisModal").close();
}

async function testRedisConnection() {
  const url = document.getElementById("redisTestUrl").value.trim() || "redis://localhost:6379/0";
  const resultEl = document.getElementById("redisTestResult");
  resultEl.innerHTML = `<span class="text-white/50">Testing ${escapeHtml(url)}…</span>`;
  try {
    const res = await fetch("/api/redis/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ redis_url: url }),
    });
    const data = await res.json();
    resultEl.innerHTML = `<span class="${data.success ? "up" : "down"}">${escapeHtml(data.message)}</span>`;
    if (data.success) loadStats();
  } catch (err) {
    resultEl.innerHTML = `<span class="down">${escapeHtml(err.message)}</span>`;
  }
}

// ---------------- Live prices (WebSocket /ws/prices) ----------------

let priceSocket = null;
const quotes = {};
const assetClass = {};

function formatPrice(q) {
  const p = q.price;
  let d = 2;
  if (q.class === "forex") d = q.symbol.includes("JPY") ? 3 : 5;
  else if (p < 1) d = 5;
  else if (p < 10) d = 4;
  return p.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
}

function formatChange(q) {
  if (q.change_pct === null || q.change_pct === undefined) return "";
  return `${q.change_pct >= 0 ? "+" : ""}${q.change_pct.toFixed(2)}%`;
}

function chgClass(q) {
  if (!q || q.change_pct === null || q.change_pct === undefined) return "pill-flat";
  return q.change_pct > 0 ? "pill-up" : q.change_pct < 0 ? "pill-down" : "pill-flat";
}

function tabClassOf(cls) {
  return ["forex", "stock", "crypto"].includes(cls) ? cls : "other";
}

function renderTickerSkeleton(assets) {
  // Two identical copies make the slide loop seamlessly.
  const one = assets.map(a => `
    <span class="ticker-item inline-flex gap-2" data-sym="${escapeHtml(a.symbol)}">
      <span class="text-white/80">${escapeHtml(a.symbol)}</span>
      <span class="val text-white/40">—</span>
      <span class="chg"></span>
    </span>`).join("");
  document.getElementById("tickerItems").innerHTML = one + one;
}

function initMarketTabs() {
  document.querySelectorAll("#marketTabs .tab").forEach(t => t.addEventListener("click", () => {
    document.querySelectorAll("#marketTabs .tab").forEach(x => x.classList.remove("tab-active"));
    t.classList.add("tab-active");
    state.marketClass = t.dataset.cls;
    renderMarketTable();
  }));
}

function renderMarketTable() {
  const rows = Object.keys(assetClass).filter(s => tabClassOf(assetClass[s]) === state.marketClass);
  document.getElementById("marketTable").innerHTML = rows.map(s => {
    const q = quotes[s];
    const chg = q ? formatChange(q) : "";
    return `
      <tr class="border-white/5 hover cursor-pointer" data-sym="${escapeHtml(s)}" onclick="setQuickSearch('${escapeHtml(s.split("/")[0])}')">
        <td class="px-0 text-white/80">${escapeHtml(s)}</td>
        <td class="val text-right text-white/60">${q ? formatPrice(q) : "—"}</td>
        <td class="w-20 px-0 text-right"><span class="chg inline-block min-w-[3.75rem] rounded-md px-1.5 py-0.5 text-center text-xs ${chgClass(q)}">${chg}</span></td>
      </tr>`;
  }).join("");
}

function applyQuotes(list) {
  list.forEach(q => {
    const prev = quotes[q.symbol];
    quotes[q.symbol] = q;
    const flash = prev && prev.price !== q.price ? (q.price > prev.price ? "tick-up" : "tick-down") : null;
    document.querySelectorAll(`[data-sym="${CSS.escape(q.symbol)}"]`).forEach(el => {
      const valEl = el.querySelector(".val");
      const chgEl = el.querySelector(".chg");
      valEl.textContent = formatPrice(q);
      chgEl.textContent = formatChange(q);
      if (el.closest("#marketTable")) {
        chgEl.classList.remove("pill-up", "pill-down", "pill-flat");
        chgEl.classList.add(chgClass(q));
      } else {
        chgEl.classList.toggle("up", q.change_pct >= 0);
        chgEl.classList.toggle("down", q.change_pct < 0);
      }
      if (flash) {
        valEl.classList.remove("tick-up", "tick-down");
        void valEl.offsetWidth; // restart the animation
        valEl.classList.add(flash);
      }
    });
  });
}

async function initLivePrices() {
  try {
    const res = await fetch("/api/assets");
    const data = await res.json();
    data.assets.forEach(a => { assetClass[a.symbol] = a.class; });
    renderTickerSkeleton(data.assets);
    renderMarketTable();
    applyQuotes(data.assets.filter(a => a.quote).map(a => a.quote));
  } catch (e) {
    console.warn("Could not load assets", e);
  }
  connectPriceSocket(1000);
}

function connectPriceSocket(retryMs) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  priceSocket = new WebSocket(`${proto}://${location.host}/ws/prices`);
  priceSocket.onopen = () => { retryMs = 1000; };
  priceSocket.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.quotes) applyQuotes(msg.quotes);
  };
  priceSocket.onclose = () => {
    setTimeout(() => connectPriceSocket(Math.min(retryMs * 2, 30000)), retryMs);
  };
}

// ---------------- Helpers ----------------

function showToast(msg) {
  const container = document.getElementById("toastContainer");
  const el = document.createElement("div");
  el.className = "alert glass rounded-xl border border-white/10 bg-black/80 py-2 text-sm shadow-xl";
  el.textContent = "✓ " + msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

function formatTimeAgo(isoString) {
  if (!isoString) return "just now";
  const seconds = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000);
  if (isNaN(seconds)) return isoString;
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
