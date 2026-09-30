# Teaching the Search Engine — Complete Guide

> From "what is an inverted index?" to distributed sharding, neural reranking, and production serving.  
> Every piece of math is derived. Every parameter has a measured reason. Every line of code is explained.

---

## Table of Contents

1. [Bird's-Eye Architecture](#1-birds-eye-architecture)
2. [Text Processing — Tokenizer & Porter Stemmer](#2-text-processing--tokenizer--porter-stemmer)
3. [The Inverted Index — v0 (Pure Python)](#3-the-inverted-index--v0-pure-python)
4. [BM25 Ranking — The Core Scoring Function](#4-bm25-ranking--the-core-scoring-function)
5. [V1 Index — NumPy Arrays & Precomputed Impacts](#5-v1-index--numpy-arrays--precomputed-impacts)
6. [MaxScore Algorithm — Exact Top-k Without Reading Everything](#6-maxscore-algorithm--exact-top-k-without-reading-everything)
7. [V2 Searcher — Native C MaxScore Kernel](#7-v2-searcher--native-c-maxscore-kernel)
8. [Index Compression — Block-Compressed Format (V3)](#8-index-compression--block-compressed-format-v3)
9. [V4 Indexer — Parallel Native Scanner & Vectorized Merge](#9-v4-indexer--parallel-native-scanner--vectorized-merge)
10. [Dense Retrieval — Bi-Encoder & Embeddings](#10-dense-retrieval--bi-encoder--embeddings)
11. [Product Quantization (PQ) — 24x Memory Compression](#11-product-quantization-pq--24x-memory-compression)
12. [Hybrid Search — BM25 + Dense Fusion](#12-hybrid-search--bm25--dense-fusion)
13. [Cross-Encoder Reranking — Third Cascade Stage](#13-cross-encoder-reranking--third-cascade-stage)
14. [IVF Index — Approximate Nearest Neighbour](#14-ivf-index--approximate-nearest-neighbour)
15. [Phrase Search](#15-phrase-search)
16. [Spell Correction — Norvig's Algorithm + Corpus Prior](#16-spell-correction--norvigs-algorithm--corpus-prior)
17. [Autocomplete / Query Suggestion](#17-autocomplete--query-suggestion)
18. [PageRank — Link-Graph Signal](#18-pagerank--link-graph-signal)
19. [Web Crawler](#19-web-crawler)
20. [Distributed Search — Sharding & Replication](#20-distributed-search--sharding--replication)
21. [Live Index — Segments, Deletes & Tiered Merges](#21-live-index--segments-deletes--tiered-merges)
22. [HTTP Server & Search UI](#22-http-server--search-ui)
23. [Forex Application Layer](#23-forex-application-layer)
24. [Benchmarking & Quality Measurement](#24-benchmarking--quality-measurement)
25. [Libraries & Tools Reference](#25-libraries--tools-reference)
26. [Parameters Encyclopedia](#26-parameters-encyclopedia)
27. [Data Flow: End-to-End Query Lifecycle](#27-data-flow-end-to-end-query-lifecycle)
28. [Index Files Reference](#28-index-files-reference)

---

## 1. Bird's-Eye Architecture

The repo contains **two layers** on top of one codebase:

```
searchengine/          ← the search engine proper
forex_app/             ← a domain application built on top of it
bench/                 ← benchmarks and quality measurements
experiments/           ← ablation studies for design decisions
```

### Search Engine Components (build order = bottom to top)

```
┌──────────────────────────────────────────────────────┐
│                      HTTP Server                      │  server.py
│  (BM25 / Hybrid / Phrase / Spell / Autocomplete UI)  │  ui.py
├──────────────────────────────────────────────────────┤
│            Hybrid Searcher (BM25 + Dense)             │  search_hybrid.py
│              Cross-Encoder Reranker (opt.)            │  cross_encoder.py
├────────────────────┬─────────────────────────────────┤
│  Lexical Index     │    Dense Index                   │
│  (v1/v2/v3)        │    (exact f32 / PQ codes)        │  dense.py, pq.py
├────────────────────┴─────────────────────────────────┤
│  Bi-Encoder (MiniLM / GTE / BGE / E5 via ONNX RT)    │  encoder.py
├──────────────────────────────────────────────────────┤
│  Tokenizer → Porter Stemmer → Postings                │  tokenizer.py, porter.py
└──────────────────────────────────────────────────────┘
```

### Query pipeline (one user query):

```
Query string
    │
    ▼
Tokenize → Stem
    │
    ├──► BM25 Searcher (C MaxScore kernel) → top-1000 candidates
    │         │
    │         ▼
    │    Bi-Encoder (encodes query vector)
    │         │
    │         ▼
    │    PQ ADC scoring (candidate re-rank) ────► blend (α · BM25 + (1-α) · dense)
    │         │                                        │
    │         └────── optional Cross-Encoder ──────────┘
    │
    ▼
top-k results → HTTP JSON response
```

---

## 2. Text Processing — Tokenizer & Porter Stemmer

### 2.1 Tokenizer

**File:** [`searchengine/tokenizer.py`](searchengine/tokenizer.py)

```python
import re
_TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
```

**What it does:**
- Lowercases the entire string (`text.lower()`)
- Extracts all contiguous runs of `[a-z0-9]` using a single compiled regex
- Discards punctuation, whitespace, special characters

**Why this design:**
- A compiled regex (`re.compile`) is ~3x faster than inline `re.findall` because the pattern is compiled once at import time
- Lowercase normalization ensures `"Federal"` and `"federal"` map to the same token
- No stopword removal at this stage — the BM25 IDF weight naturally down-weights common words because they appear in almost every document, giving them near-zero IDF score

**Example:**
```
Input:  "The Federal Reserve's interest-rate policy (2024)"
Output: ["the", "federal", "reserve", "s", "interest", "rate", "policy", "2024"]
```

---

### 2.2 Porter Stemmer

**File:** [`searchengine/porter.py`](searchengine/porter.py)

The Porter stemmer (M.F. Porter, 1980) reduces inflected forms to a common root so `"running"`, `"runs"`, `"ran"` all map to `"run"`.

#### Core Concept: VC sequences

A word is measured by the number of **vowel-consonant (VC) sequences** it contains — this quantity `m` governs when a rule fires.

```python
def _m(w: str) -> int:
    """number of VC sequences in w."""
    n = 0
    i = 0
    # skip leading consonants
    while i < ln and _cons(w, i):
        i += 1
    while i < ln:
        # skip vowel run
        while i < ln and not _cons(w, i):
            i += 1
        if i == ln:
            break
        n += 1          # one VC unit completed
        # skip consonant run
        while i < ln and _cons(w, i):
            i += 1
    return n
```

**Examples:**
```
"tree"     = TR·EE         → m=0  (no VC after the C run)
"trees"    = TR·EE·S       → m=1
"troubles" = TR·O·UBL·E·S  → m=2
```

#### The 5 Steps

| Step | Purpose | Example |
|------|---------|---------|
| 1a | Plural `-s` | `caresses→caress`, `ponies→poni`, `cats→cat` |
| 1b | Past tense `-ed`, `-ing` | `plastered→plaster`, `motoring→motor` |
| 1c | `-y` → `-i` when there's a vowel in stem | `happy→happi` |
| 2 | Suffix substitution (longer suffixes) | `ational→ate`, `tional→tion` |
| 3 | Suffix substitution (shorter suffixes) | `icate→ic`, `alize→al` |
| 4 | Remove derivational suffixes | `revival→reviv` (m>1) |
| 5a | Remove trailing `e` | `probate→probat` (m>1) |
| 5b | Remove double consonant `ll` | `controll→control` (m>1) |

#### Why stemming matters for quality

The Anserini baseline on MS MARCO achieves **MRR@10 = 0.184 with Porter stemming** vs lower without. This engine's measured result confirms that: stemmed tuned = **0.1892**, unstemmed default = **0.186**.

---

## 3. The Inverted Index — v0 (Pure Python)

**File:** [`searchengine/indexer.py`](searchengine/indexer.py)

This is the simplest possible search index — useful for understanding the concept before optimizations.

### Data Structure

```python
class IndexV0:
    postings: dict[str, list[tuple[int, int]]]
    # term → [(docid, term_frequency), ...]

    doc_lens: list[int]   # doc_lens[docid] = number of tokens in that doc
    doc_ids: list[int]    # doc_ids[docid] = external passage ID (MS MARCO pid)
    n_docs: int
    total_len: int        # sum of all doc_lens (for computing avgdl)
```

### Adding a Document

```python
def add(self, ext_id: int, text: str) -> None:
    toks = tokenize(text)
    docid = self.n_docs           # internal 0-based counter
    tf: dict[str, int] = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1  # count term frequency per document
    for t, f in tf.items():
        self.postings[t].append((docid, f))   # one entry per (term, doc) pair
    self.doc_lens.append(len(toks))
    self.doc_ids.append(ext_id)
    self.n_docs += 1
    self.total_len += len(toks)
```

**Key insight:** Only ONE entry per term per document — even if "bank" appears 5 times, we store `("bank", 5)` not five entries. This is called **tf-aggregation**.

### Input Format

Documents come from a **TSV file** (tab-separated values):
```
pid<TAB>passage text
```
Example:
```
7187744	The interest rate set by the Federal Reserve is 5.25%.
```

### Why pickle for persistence (v0)?

Python's `pickle` serializes arbitrary Python objects to bytes. It's naive (human-unreadable, no streaming, loads everything into RAM) but trivially correct. v1 replaces this with memory-mapped NumPy arrays.

---

## 4. BM25 Ranking — The Core Scoring Function

**File:** [`searchengine/search.py`](searchengine/search.py)

BM25 (Best Match 25, Robertson & Ogilvie 1994, later Robertson et al. 2009) is the industry-standard term-based relevance model.

### The Formula

$$\text{Score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t,D) \cdot (k_1 + 1)}{f(t,D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
| Symbol | Meaning | Value in this engine |
|--------|---------|----------------------|
| $f(t, D)$ | Term frequency of $t$ in document $D$ | Stored per posting |
| $\|D\|$ | Length of document $D$ in tokens | `doc_lens[docid]` |
| $\text{avgdl}$ | Average document length across all docs | `total_len / n_docs` |
| $k_1$ | TF saturation — controls diminishing returns | **0.82** (tuned, default 0.9) |
| $b$ | Length normalization strength (0=off, 1=full) | **0.75** (tuned, default 0.4) |
| $\text{IDF}(t)$ | Inverse document frequency | See below |

### IDF Formula (Robertson-Spärck Jones variant)

$$\text{IDF}(t) = \log\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right)$$

Where:
- $N$ = total number of documents
- $\text{df}(t)$ = number of documents containing term $t$

The `+0.5` smoothing prevents division by zero and gives a sensible score even for extremely common terms.

```python
idf = math.log(1.0 + (idx.n_docs - df + 0.5) / (df + 0.5))
```

### TF Saturation — Why k1 matters

Without saturation, a document with "bank" 100 times would score 100× higher than one with "bank" once, even if the extra occurrences add no information.

BM25's TF component saturates:

$$\text{TF-part}(f, |D|) = \frac{f \cdot (k_1 + 1)}{f + k_1 \cdot \text{norm}(|D|)}$$

As $f \to \infty$, this approaches $k_1 + 1$. So **the maximum possible TF contribution is $k_1 + 1$**, regardless of raw count.

- `k1 = 0`: No TF weighting at all (binary model)
- `k1 = 1.2`: Moderate saturation (classic TREC default)
- `k1 = 0.82`: Faster saturation (tuned for MS MARCO short passages)

### Length Normalization — Why b matters

$$\text{norm}(|D|) = 1 - b + b \cdot \frac{|D|}{\text{avgdl}}$$

- If `b = 0`: Length normalization is disabled; long documents score the same as short ones.
- If `b = 1`: Full normalization; scoring is purely density-based.
- If `b = 0.75` (tuned value): A document twice the average length has its TF contribution halved, approximately.

### The Accumulation Loop (v0)

```python
scores: dict[int, float] = {}
for t in set(terms):          # deduplicate query terms
    plist = idx.postings.get(t)
    if not plist:
        continue
    df = len(plist)
    idf = math.log(1.0 + (idx.n_docs - df + 0.5) / (df + 0.5))
    for docid, tf in plist:
        dl = doc_lens[docid]
        s = idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl))
        scores[docid] = scores.get(docid, 0.0) + s

# Retrieve top-k using a heap (O(n log k) instead of O(n log n) sort)
top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
```

**`heapq.nlargest` is O(n log k)** — much faster than sorting all scores when k << n.

---

## 5. V1 Index — NumPy Arrays & Precomputed Impacts

**Files:** [`searchengine/indexer_v4.py`](searchengine/indexer_v4.py), [`searchengine/search_v1.py`](searchengine/search_v1.py)

The v0 approach has two critical problems at scale:
1. Python dict lookups are slow (hash table overhead per posting)
2. Computing BM25 during every query is redundant

V1 solves both with **precomputed impacts stored in memory-mapped NumPy arrays**.

### Precomputing Impacts

The BM25 score for a posting `(t, d)` depends only on:
- IDF of `t` (a corpus-level constant)
- TF of `t` in `d` (stored per posting)
- `doc_lens[d]` (stored per document)
- `avgdl` (a corpus-level constant)

Since all these are known at index time, we can compute the score once and store it:

```python
# In indexer_v4.py — vectorized impact computation
idf_t = np.log(1.0 + (n_docs - gdfs + 0.5) / (gdfs + 0.5))  # per-term IDF
impacts = idf_p * tf_f * (k1 + 1.0) / (
    tf_f + k1 * (1.0 - b + b * dl[docids] / avgdl))
```

At query time, scoring becomes pure array addition:

```python
scores[self.docids[s:e]] += self.impacts[s:e]
```

This is **vectorized NumPy addition** — the inner loop runs in C, not Python.

### Index Layout on Disk

```
index_dir/
├── terms.pkl           # dict: stemmed_term → term_id (integer)
├── offsets.u64.npy     # uint64[n_terms+1] — slice boundaries in docids/impacts
├── docids.u32.npy      # uint32[total_postings] — docid per posting
├── impacts.f32.npy     # float32[total_postings] — precomputed BM25 score
├── tfs.u8.npy          # uint8[total_postings] — raw term frequency (clamped ≤63)
├── doclens.u32.npy     # uint32[n_docs] — token count per doc
├── max_impact.f32.npy  # float32[n_terms] — max impact per term (for MaxScore)
└── meta.json           # {n_docs, n_terms, avgdl, k1, b, stemmed: true}
```

### How the Offset Array Works

Given term ID `tid`, the postings for that term live at:
```python
s = offsets[tid]      # start (inclusive)
e = offsets[tid + 1]  # end (exclusive)
docids_slice = docids[s:e]
impacts_slice = impacts[s:e]
```

This is a **Compressed Sparse Row (CSR)** layout — the standard for sparse matrices. One integer lookup gives the full posting list, no pointers needed.

### Adaptive Top-k Selection

```python
ADAPTIVE_THRESHOLD = 1_000_000

if total > ADAPTIVE_THRESHOLD:
    # More than 1M postings touched: full-array partition is faster
    idx = np.argpartition(scores, -k_)[-k_:]
else:
    # Few touched: only examine unique touched docids
    touched = np.unique(np.concatenate([docids[s:e] for s, e in slices]))
    vals = scores[touched]
    idx = np.argpartition(vals, -k_)[-k_:]
```

`np.argpartition` is O(n) (quickselect algorithm) vs O(n log n) for full sort. It finds the top-k indices without sorting everything.

---

## 6. MaxScore Algorithm — Exact Top-k Without Reading Everything

**Concept paper:** Turtle & Flood (1995), formalized in PISA (Mallia et al.)  
**File:** [`searchengine/native/maxscore.c`](searchengine/native/maxscore.c)

The problem with the v1 accumulation loop: for a common query like `"the bank"`, the posting list for `"the"` might contain 5 million documents. We read all 5M even though only 10 will appear in the result.

**MaxScore** allows skipping documents that *cannot possibly* enter the top-k.

### Key Insight: Upper Bound on a Document's Score

For a document `d`, its maximum possible BM25 score is bounded by summing the max impact of every query term:

$$\text{UB}(d) \leq \sum_{t \in Q} \text{maxImpact}(t)$$

But if `d` is missing from some terms' posting lists, those terms contribute 0. We can tighten this:

$$\text{UB}(d) = \text{score\_from\_seen\_terms}(d) + \sum_{t \in Q \setminus \text{seen}(d)} \text{maxImpact}(t)$$

### Essential vs. Non-Essential Lists

Sort terms by their `maxImpact` ascending. Define a threshold `θ` = the k-th best score seen so far.

A term `t` is **non-essential** if even adding its full `maxImpact` to the current threshold cannot change who is in the top-k:

$$\text{prefixSum}[i] = \sum_{j=0}^{i} \text{maxImpact}[j]$$

If `prefixSum[pivot-1] <= θ`, then lists `0..pivot-1` are non-essential. Documents missing from all essential lists cannot enter the top-k, so we skip them entirely.

### The Algorithm

```c
// 1. Sort terms by max_impact ASCENDING
// 2. Maintain a min-heap of size k (threshold θ = heap top)
// 3. Iterate through "candidate" docids from essential lists only
// 4. For each candidate d:
//    a. Score essential lists (they all contain d or adjacent)
//    b. For non-essential lists: only probe IF remaining UB > θ
//    c. If final score > θ: insert into heap, update θ

for (;;) {
    // find the next candidate: min docid among essential lists
    uint32_t d = UINT32_MAX;
    for (int i = pivot; i < nterms; i++)
        if (lists[i].pos < lists[i].len && lists[i].ids[lists[i].pos] < d)
            d = lists[i].ids[lists[i].pos];
    if (d == UINT32_MAX) break;

    // score essential lists
    double score = 0.0;
    for (int i = pivot; i < nterms; i++) { /* accumulate */ }

    // probe non-essential lists (high-impact ones first)
    for (int i = pivot - 1; i >= 0; i--) {
        if (score + prefix[i] <= theta) break;  // pruning condition
        gallop(&lists[i], d);                   // fast-forward to d
        if (at d) score += lists[i].imp[pos];
    }

    // heap update
    if (score > theta || heap not full) { ... }
}
```

### Galloping Search

When we need to fast-forward a posting list to docid `d`, we use **galloping** (also called exponential search):

```c
// Double the step size until we overshoot, then binary search
int64_t step = 1, last = pos;
while (pos + step < len && ids[pos + step] < d) {
    last = pos + step;
    step <<= 1;   // double the step
}
// binary search in (last, pos+step]
```

This is O(log(gap)) where gap is the distance to `d`, much better than linear scan for sparse lists.

### Measured Savings

On a typical MS MARCO query, MaxScore reads **10–40×** fewer postings than exhaustive evaluation, while returning **exactly the same results** (modulo floating-point addition order).

---

## 7. V2 Searcher — Native C MaxScore Kernel

**Files:** [`searchengine/search_v2.py`](searchengine/search_v2.py), [`searchengine/native/maxscore.c`](searchengine/native/maxscore.c), [`searchengine/native/__init__.py`](searchengine/native/__init__.py)

V2 is the Python wrapper that calls the MaxScore C kernel via `ctypes`.

### Why ctypes?

- Python's GIL (Global Interpreter Lock) is released during C extension calls
- The inner scoring loop runs at native C speed (~100× faster than equivalent Python)
- No compilation step for the user — `ctypes.CDLL` loads a pre-compiled `.dylib`

### Per-Thread Scratch Buffers

```python
class _Scratch(threading.local):
    """Per-thread ctypes buffers — the kernel releases the GIL,
    so concurrent searches would race on shared scratch."""

    def __init__(self):
        self.ids_ptrs = (ctypes.c_void_p * MAX_TERMS)()
        self.imp_ptrs = (ctypes.c_void_p * MAX_TERMS)()
        self.lens = (ctypes.c_int64 * MAX_TERMS)()
        self.maxs = (ctypes.c_float * MAX_TERMS)()
        self.out_ids = (ctypes.c_uint32 * 1024)()
        self.out_scores = (ctypes.c_float * 1024)()
        self.stats = (ctypes.c_int64 * 2)()
```

`threading.local` gives each thread its own copy of the scratch buffers — avoids data races without locking.

### Passing Memory to C

```python
# The C kernel needs raw pointers into the NumPy mmap'd arrays
self._ids_base = self.docids.ctypes.data   # base address of docids array
self._imp_base = self.impacts.ctypes.data  # base address of impacts array

# Compute pointer for term tid's posting list:
t.ids_ptrs[i] = self._ids_base + 4 * s    # uint32 = 4 bytes, start at offset s
t.imp_ptrs[i] = self._imp_base + 4 * s    # float32 = 4 bytes
```

### The `intersect` Method

```python
def intersect(self, terms: list[str], max_out: int = 200_000) -> np.ndarray:
    """All docids containing EVERY given term — for phrase search."""
    spans.sort()  # smallest df first: result can't be bigger than smallest list
    cand = np.asarray(self.docids[s:e])    # start with smallest list
    for _, s, e in spans[1:]:
        arr = self.docids[s:e]
        idx = np.searchsorted(arr, cand)   # binary search each candidate
        idx[idx == len(arr)] = len(arr) - 1
        cand = cand[arr[idx] == cand]      # keep only those found in arr
    return cand
```

`np.searchsorted` performs binary search — O(|cand| × log(|arr|)) total.

---

## 8. Index Compression — Block-Compressed Format (V3)

**Files:** [`searchengine/compress_index.py`](searchengine/compress_index.py), [`searchengine/native/compress.c`](searchengine/native/compress.c), [`searchengine/search_v3.py`](searchengine/search_v3.py), [`searchengine/native/bmw.c`](searchengine/native/bmw.c)

The v1 flat format stores docids as raw `uint32` — 4 bytes per posting regardless of the actual value. Block compression cuts this by 3–4×.

### Why delta coding?

Posting lists store docids in **ascending order** (because documents are added in order). Consecutive docids differ by small amounts:

```
Actual docids:  [1042, 1043, 1089, 1091, 1092, 1200, ...]
Deltas (d_i - d_{i-1} - 1):
                [1041,    0,   45,    1,    0,  107, ...]
```

Small values need fewer bits. We then pack them with **bit-width = bits needed for the maximum delta in the block**.

### Block Structure (128 postings per block)

```
Block:
  [ceil(c×w/8) bytes]   LSB-first bitpacked deltas (w = bits of max delta)
  [c bytes]             u8 quantized impacts (straight copy)
```

```c
// In compress.c
uint8_t w = (uint8_t)width_of(maxv);  // bits needed for max delta in block
// bitpack all deltas into w bits each
```

### Skip Arrays (one entry per block)

| Array | Type | Content |
|-------|------|---------|
| `block_last.u32.npy` | uint32 | Last docid in each block |
| `block_width.u8.npy` | uint8 | Bit width of each block |
| `block_maxq.u8.npy` | uint8 | Max quantized impact in each block |

These arrays are tiny (a few MB) and kept **hot in cache**. The `bmw_query` kernel can check `block_last` to skip entire blocks without decompressing them.

### Impact Quantization

Float32 impacts are quantized to uint8:

```python
scale = float(imp.max()) / 255.0
impq = np.rint(np.clip(imp / scale, 0, 255)).astype(np.uint8)
```

The **same scale** must be used across all shards of a distributed index (otherwise scores from different shards are not comparable — this cost ~2.7% of top-10 agreement in experiments).

### Compression Results

| Format | Size | p50 latency | MRR@10 |
|--------|------|-------------|--------|
| v1 flat | 1× (baseline) | 0.75ms | 0.1892 |
| v3 compressed | **3.65× smaller** | ~1.7× slower | 0.1892 (−4e-5) |

The quality loss of 4e-5 MRR is below measurement noise — both formats are equivalent in practice.

### V3 Searcher — BMW (Block-Max WAND)

```python
# In search_v3.py — the C library provides bmw_query
cnt = self.lib.bmw_query(
    self._p["blob"], self._p["blast"], self._p["bwidth"],
    self._p["bmaxq"], self._p["boff"],
    t.tb0, t.tnb, t.tdf, t.tmaxq,
    n, min(k, 1024), t.out_ids, t.out_scores, t.stats)
```

The BMW kernel extends MaxScore with **block-level pruning**: when galloping to docid `d`, it checks `block_last` to find the right block without decoding, and checks `block_maxq` to decide if even the best possible posting in that block could affect the top-k. If not, the entire block is skipped.

---

## 9. V4 Indexer — Parallel Native Scanner & Vectorized Merge

**File:** [`searchengine/indexer_v4.py`](searchengine/indexer_v4.py)  
**Native:** [`searchengine/native/scanner.c`](searchengine/native/scanner.c)

The v0 indexer is single-threaded Python. V4 is a **multi-process pipeline** that achieves near-linear speedup on multi-core machines.

### Pipeline

```
collection.tsv
    │
    ▼  Step 1: byte-level file split (line-aligned)
    │           W contiguous byte ranges
    │
    ▼  Step 2: W native scanner processes (C, via subprocess)
    │           each: tokenize + Porter-stem + count TF
    │           output: binary shard files
    │
    ▼  Step 3a: read shards, build global vocabulary union
    │
    ▼  Step 3b: counting-sort placement (vectorized NumPy)
    │           map local term IDs → global term IDs
    │           place every posting at its final position
    │
    ▼  Step 4: compute BM25 impacts vectorized
    │
    ▼  Step 5: write v1-format index files
```

### File Split (line-aligned)

```python
def _split_offsets(path: str, w: int) -> list[tuple[int, int]]:
    size = os.path.getsize(path)
    cuts = [0]
    with open(path, "rb") as f:
        for i in range(1, w):
            f.seek(size * i // w)
            f.readline()      # advance past the current line so we never split mid-line
            cuts.append(f.tell())
    cuts.append(size)
    return [(cuts[i], cuts[i + 1]) for i in range(w)]
```

Each worker gets a byte range `[start, end)` and processes complete lines.

### Shard Binary Format (MAGIC = `0x53484152445F3032`)

```
Header (40 bytes):
  [0:8]   magic uint64
  [8:12]  stem_flag uint32
  [12:16] min_pid uint32
  [16:20] n_docs uint32
  [20:24] n_terms uint32
  [24:32] total_postings uint64
  [32:40] names_bytes uint64

Body:
  doclens  uint16[n_docs]
  term_lens uint16[n_terms]
  term_names bytes[names_bytes]
  dfs      uint32[n_terms]   (document frequencies)
  packed   uint32[total]     (upper 26 bits = docid, lower 6 bits = TF)
```

**TF packing:** `packed[i] = (docid << 6) | min(tf, 63)`. This stores both fields in a single 32-bit integer. TF is clamped at 63 (`TF_CLAMP = 63`) — BM25 saturates quickly, so higher values add almost nothing.

### Vectorized Counting-Sort Placement

This is the clever core of v4 — placing every posting at its final array position without a giant `argsort`:

```python
# For each shard, compute where each posting goes in the global arrays
starts = offsets[gm].astype(np.int64) + written[gm]
# tgt[k] = position in global array for posting k in this shard
tgt = np.repeat(starts, dfs)
tgt += np.arange(len(packed), dtype=np.int64) - np.repeat(local_offs[:-1], dfs)
packed_all[tgt] = packed
written[gm] += dfs
```

`np.repeat` and fancy indexing: no per-posting Python loop, the entire shard is placed in one vectorized NumPy call.

---

## 10. Dense Retrieval — Bi-Encoder & Embeddings

**File:** [`searchengine/encoder.py`](searchengine/encoder.py)

BM25 is a **lexical** model — it matches exact words. Dense retrieval uses **semantic embeddings**: queries and documents are mapped to vectors in the same high-dimensional space, and similarity is measured by dot product (cosine).

### Bi-Encoder Architecture

```
Query: "central bank policy"
         │
    [Tokenizer]
         │
    [Transformer] (MiniLM-L6-v2, 6-layer BERT variant)
         │
    [Pooling: mean of token embeddings]
         │
    [L2 Normalize]
         │
    float32[384]   ← 384-dimensional unit vector
```

The same model encodes both queries and passages, but queries are encoded online (at query time) and passages are encoded offline (at index time).

### Supported Models & Their Differences

| Config | Pool | Query Prefix | Doc Prefix |
|--------|------|-------------|------------|
| `models/minilm` | mean | (none) | (none) |
| `models/gte` | mean | (none) | (none) |
| `models/bge` | CLS | `"Represent this sentence for searching relevant passages: "` | (none) |
| `models/e5` | mean | `"query: "` | `"passage: "` |

**CLS pooling** uses only the first special token (the `[CLS]` token), which some models train to aggregate sentence meaning.  
**Mean pooling** averages all token embeddings weighted by the attention mask.

```python
if self.pool == "cls":
    emb = out[:, 0, :]                      # shape: (batch, dim)
else:
    m = am[:, :, None].astype(np.float32)   # attention mask: (batch, seq, 1)
    emb = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
```

### ONNX Runtime & int8 Quantization

The model is exported to ONNX format and loaded with ONNX Runtime:

```python
so = ort.SessionOptions()
so.intra_op_num_threads = threads
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
```

**int8 dynamic quantization** converts weight matrices from float32 to int8:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
```

This gives **2–3× speedup on ARM NEON** with ~1% quality loss — no calibration data needed since only weights (not activations) are quantized.

### Length-Bucketed Batching

MS MARCO passages average ~58 tokens but naive batching pads every batch to the batch maximum:

```python
# Without bucketing: if one passage has 200 tokens and 63 others have 30,
# the entire batch is padded to 200 → massive waste

# With bucketing: sort by length, so similar-length passages batch together
order = np.argsort([len(x) for x in id_lists], kind="stable")
```

This removes most padding waste and speeds encoding by ~2×.

---

## 11. Product Quantization (PQ) — 24x Memory Compression

**File:** [`searchengine/pq.py`](searchengine/pq.py)

The MS MARCO corpus has **8.84M documents × 384 dimensions × 4 bytes = 13.6 GB** of float32 vectors. This doesn't fit on a typical machine. PQ compresses it to **566 MB** (24×) with ~0.8% quality loss.

### The Idea

Split each 384-dim vector into `M = 64` **subvectors** of `dsub = 384/64 = 6` dimensions each.

For each subspace, cluster all document subvectors into `256` centroids using k-means. Then represent each document's subvector by the index (0–255) of its nearest centroid.

```
Original vector (384 floats = 1536 bytes):
  [x0, x1, x2, x3, x4, x5 | x6, x7, ..., x11 | ... | x378, ..., x383]
   ←── subspace 0 ──────→   ←── subspace 1 ──────→       ←── sub 63 ──→

Compressed code (64 bytes):
  [c0 | c1 | ... | c63]
   each c_i ∈ {0..255} = index of nearest centroid in subspace i
```

### K-Means Training (Lloyd's Algorithm)

```python
def _kmeans(x: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(n, k, replace=False)].copy()   # random init
    for _ in range(iters):
        # Assign step: argmin over ||x - c||^2 = argmin over -2<x,c> + ||c||^2
        cn = (cent * cent).sum(1)
        assign = np.argmin(-2.0 * (x @ cent.T) + cn[None, :], axis=1)
        # Update step: new centroid = mean of assigned points
        np.add.at(sums, assign, x)
        cent[nz] = sums[nz] / counts[nz, None]
        # Re-seed dead centroids (clusters that lost all points)
        cent[~nz] = x[rng.choice(n, ...)]
    return cent
```

The distance formula trick: $\|x - c\|^2 = \|x\|^2 - 2\langle x, c \rangle + \|c\|^2$. Since $\|x\|^2$ is constant per point, we only need $-2 x^\top C + \|c\|^2$ for the argmin.

### Asymmetric Distance Computation (ADC)

At query time, the query stays in full float32. For each subspace, we precompute a lookup table of dot products between the query subvector and all 256 centroids:

```python
def lut(self, q: np.ndarray) -> np.ndarray:
    """Returns (M, 256) table."""
    t = np.empty((self.m, 256), np.float32)
    for i in range(self.m):
        t[i] = self.centroids[i] @ q[i * self.dsub:(i + 1) * self.dsub]
    return t
```

Scoring a document is then just `M` table lookups and additions:

```python
@staticmethod
def adc(lut: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Scores for codes (n, m) — no decompression needed."""
    return lut[np.arange(lut.shape[0])[None, :], codes].sum(1)
```

**Why "asymmetric"?** We never quantize the query — only documents are quantized. Asymmetric ADC is more accurate than symmetric scoring (both quantized) because the query's approximation error is zero.

---

## 12. Hybrid Search — BM25 + Dense Fusion

**File:** [`searchengine/search_hybrid.py`](searchengine/search_hybrid.py)

### The Problem with Each Approach Alone

| Approach | Strength | Weakness |
|----------|---------|---------|
| BM25 | Exact term matching, very fast | Misses synonyms, paraphrases |
| Dense | Semantic similarity, handles vocabulary mismatch | Expensive, sensitive to domain |

Hybrid search combines both.

### Pipeline

```python
def search(self, query: str, k: int = 10, explain: bool = False):
    # 1. BM25 retrieval: top-K candidates (K = 1000)
    hits = self.bm25.search(query, self.topk_bm25)

    # 2. Encode query (once, full precision)
    qv = self.enc.encode([query], batch=1, is_query=True)[0]

    # 3. ADC score candidates with PQ codes
    pids = np.array([p for p, _ in hits], np.int64)
    bs = np.array([s for _, s in hits], np.float32)
    ds = PQ.adc(self.pq.lut(qv), np.asarray(self.codes[pids]))

    # 4. Normalize both scores to [0,1] and blend
    blend = self.alpha * _minmax(bs) + (1.0 - self.alpha) * _minmax(ds)

    # 5. Return top-k by blended score
    order = np.argsort(-blend)[:k]
```

### Min-Max Normalization

$$\hat{s} = \frac{s - \min(s)}{\max(s) - \min(s)}$$

This maps the score distribution to [0, 1] so BM25 scores (raw BM25 floats) and dense scores (cosine similarities ≈ [-1, 1]) are on the same scale before blending.

### Alpha — The Fusion Weight

$$\text{blend} = \alpha \cdot \hat{s}_{\text{BM25}} + (1-\alpha) \cdot \hat{s}_{\text{dense}}$$

| Alpha | Regime | MRR@10 (MS MARCO) |
|-------|--------|-------------------|
| 0.0 | Pure dense | — |
| 0.1 | Mostly dense | **0.315** (in-domain best) |
| 0.5 | Equal blend | 0.253 |
| 1.0 | Pure BM25 | 0.189 |

**Critical insight:** The right alpha depends on the corpus. On out-of-domain BEIR SciFact, alpha=0.5 (nDCG@10 0.708) is better than alpha=0.1 (0.611). For an unknown corpus, start at 0.5.

### Retrieval Depth K — Recall Ceiling

| K | Recall@K | MRR@10 | p99 |
|---|----------|--------|-----|
| 50 | 0.61 | 0.296 | ~25ms |
| 200 | 0.75 | 0.313 | ~32ms |
| 500 | 0.83 | 0.318 | ~38ms |
| **1000** | **0.87** | **0.320** | **47ms** |

K=1000 is chosen: a relevant passage BM25 misses at depth K can **never** be recovered by reranking. ADC scoring is cheap (table lookups), so going deeper costs only BM25 traversal time.

---

## 13. Cross-Encoder Reranking — Third Cascade Stage

**File:** [`searchengine/cross_encoder.py`](searchengine/cross_encoder.py)

The bi-encoder encodes query and document independently. A **cross-encoder** takes `[CLS] query [SEP] document [SEP]` as a single input and outputs a relevance score — full attention between query and document tokens.

### Quality vs Speed

| Stage | MRR@10 | p50 latency |
|-------|--------|-------------|
| BM25 only | 0.189 | 0.75ms |
| BM25 + bi-encoder | 0.311 | ~45ms |
| BM25 + bi-encoder + cross-encoder | **0.388** (+25%) | **141ms** |

The cross-encoder adds +25% MRR but exceeds the p99 < 100ms goal. It is an **opt-in tier** — never the default, activated by the user with `?rerank=ce`.

### Why depth=20 for cross-encoder?

The cross-encoder is evaluated on only the top-`depth` hybrid results:

| depth | MRR@10 (train) |
|-------|----------------|
| 10 | 0.4046 |
| **20** | **0.4151** |
| 50 | 0.4077 |

The optimum is non-monotone because the int8-quantized cross-encoder gets noisy at the margins. depth=20 is derived by measuring on the TRAIN split only — never on dev.

---

## 14. IVF Index — Approximate Nearest Neighbour

**File:** [`searchengine/ann.py`](searchengine/ann.py)

**IVF = Inverted File Index** — used for pure dense search (without BM25 as a first stage).

### Why Not HNSW?

| Method | Memory | Quality | Fits this machine? |
|--------|--------|---------|-------------------|
| HNSW | Several GB of graph links | Excellent | ❌ (16GB machine with 764MB lexical + 849MB codes) |
| IVFPQ | Assignment array (4B/doc) + centroids | Good | ✅ |

### Construction

1. **Train coarse centroids:** k-means on a sample of PQ-reconstructed vectors → `n_clusters = 4096` centroids
2. **Assign every document** to its nearest centroid (chunked to avoid 13.6GB peak memory)
3. **Build CSR inverted lists:** sorted docids grouped by cluster

```python
order = np.argsort(assign, kind="stable").astype(np.int32)  # sort by cluster
counts = np.bincount(assign, minlength=n_clusters)
offsets = np.cumsum(np.concatenate([[0], counts]))           # CSR offsets
```

### Query

```python
def search(self, qv: np.ndarray, k: int = 100, nprobe: int = 32):
    cs = self.centroids @ qv                                    # cosine to all centroids
    probe = np.argpartition(-cs, nprobe)[:nprobe]              # top nprobe centroids
    cand = np.concatenate([postings[offsets[c]:offsets[c+1]] for c in probe])
    cand = np.sort(cand)                                        # sorted → sequential mmap
    scores = PQ.adc(self.pq.lut(qv), codes[cand])
    top = np.argpartition(-scores, k-1)[:k]
```

`np.sort(cand)` before mmap access is critical: sorted access makes the OS prefetch sequential pages efficiently; random access causes many page faults.

---

## 15. Phrase Search

**File:** [`searchengine/phrase.py`](searchengine/phrase.py)

When a user queries `"cost of living"` (with quotes), they want exact phrase matches, not just documents containing all three words in any order.

### Parse Phase

```python
# Input: 'best "machine learning" resources "neural networks"'
# phrases = [["machine", "learning"], ["neural", "networks"]]
# loose = ["best", "resources"]
```

### Fast Path: Ranked Walk

```python
for depth in (100, 500, 2500, 10_000):
    ranked = self.s.search(full_query, depth)
    for (pid, sc), text in zip(ranked, texts):
        if all(r.search(text) for r in regexes):   # regex checks adjacency
            hits.append((pid, sc))
            if len(hits) >= k:
                break
```

Walk the BM25 ranking from best to worst. For each document, verify the phrase using a compiled regex. Stop once k verified hits are found. Because BM25 already ranks relevant documents first, this typically verifies only a few dozen before finding k.

### Slow Path: Exhaustive Intersection

When the ranked prefix runs out (rare phrases, common co-occurrences):

```python
cands = self.s.intersect(sorted(set(all_terms)), max_out=200_000)
# then scan all candidates for phrase regex matches
```

`intersect` uses `np.searchsorted` to find documents containing all phrase terms. This gives exact results by construction.

### Phrase Regex

```python
def phrase_regex(phrase: list[str]) -> re.Pattern:
    # Matches exact sequence with possible stemming variants
    # e.g. ["machine", "learning"] → \bmachine\s+learning\b
    return re.compile(r"\b" + r"\s+".join(re.escape(t) for t in phrase), re.I)
```

---

## 16. Spell Correction — Norvig's Algorithm + Corpus Prior

**File:** [`searchengine/spell.py`](searchengine/spell.py)

### Algorithm (Norvig 2007 + df prior)

```python
RARE = 5   # threshold: a term appearing in <5 docs is probably a typo

def correct_term(self, w: str) -> str | None:
    wdf = self.df.get(w, 0)
    if wdf >= RARE or len(w) < 3:
        return None              # no correction needed

    # Generate all strings at edit distance 1
    e1 = set(self._edits1(w))

    # Find the best: highest df among e1 candidates
    best, bdf = self._best(e1)
    if best is not None and bdf > max(wdf * 100, RARE):
        return best

    # Try edit distance 2 (only if no good d1 found)
    seen = set()
    for x in e1:
        for y in self._edits1(x):
            seen.add(y)
    best2, bdf2 = self._best(seen)
    if best2 is not None and bdf2 > max(wdf * 1000, RARE * 20):
        return best2
    return None
```

### Edit Operations (distance 1)

For each split `(a, b)` of word `w`:
- **Delete:** `a + b[1:]` — remove one character
- **Transpose:** `a + b[1] + b[0] + b[2:]` — swap two adjacent characters
- **Replace:** `a + c + b[1:]` — substitute one character from alphabet
- **Insert:** `a + c + b` — insert one character

Total: ~450 candidates per word (54 deletions + 52 transposes + ~52×26 replaces + ~(len+1)×36 inserts).

### The Prior: P(1 typo) >> P(2 typos)

Distance-1 corrections always win over distance-2, regardless of df. The threshold ratio (×100 for d1, ×1000 for d2) encodes this belief: a word appearing 100× more often than the query word is almost certainly the intended term.

This engine shows **"did you mean"** but never silently applies the correction — user intent is respected.

---

## 17. Autocomplete / Query Suggestion

**File:** [`searchengine/suggest.py`](searchengine/suggest.py)

### Data Source

MS MARCO train query log: **502K real user queries** sorted lexicographically.

### Implementation

```python
class Suggester:
    def __init__(self, train_queries_path: str):
        # Load and deduplicate all queries, sort lexicographically
        qs.sort()
        self.qs = qs

    def suggest(self, prefix: str, k: int = 8) -> list[str]:
        prefix = prefix.lower().strip()
        # Binary search to find the range of queries matching the prefix
        lo = bisect.bisect_left(self.qs, prefix)
        hi = bisect.bisect_right(self.qs, prefix + "\uffff")   # Unicode max char caps the range
        cands = self.qs[lo:min(hi, lo + 200)]
        cands.sort(key=len)       # shorter = more general = better suggestion
        return cands[:k]
```

**`bisect_left` / `bisect_right`** are O(log N) binary searches. The entire suggest operation takes O(log N + window_size) — essentially free.

**Ranking heuristic:** Sort by length (ascending). With no explicit query-frequency signal, shorter queries are more general and serve as better autocomplete suggestions.

---

## 18. PageRank — Link-Graph Signal

**File:** [`searchengine/pagerank.py`](searchengine/pagerank.py)

PageRank (Brin & Page 1998) measures a page's importance by the weighted sum of PageRanks of pages that link to it.

### The Formula

$$\text{PR}(u) = \frac{1 - d}{N} + d \sum_{v \to u} \frac{\text{PR}(v)}{\text{outDegree}(v)}$$

Where:
- $d$ = damping factor (typically 0.85) — probability the "random surfer" follows a link vs. teleports
- $N$ = total number of pages
- $v \to u$ = pages that link to $u$

### Power Iteration

```python
# Initialize uniformly
pr = np.ones(N) / N
for _ in range(max_iter):
    new_pr = (1 - d) / N
    # For each page v, distribute its PR to outlinks
    for v, u_list in graph.items():
        share = pr[v] / len(u_list)
        for u in u_list:
            new_pr[u] += d * share
    delta = np.abs(new_pr - pr).max()
    pr = new_pr
    if delta < tol:
        break
```

### Why PageRank is OFF by default (beta=0)

On a 7k-page web crawl, PageRank measurably **hurts** ranking (measured in experiments/). The crawl is too small and too biased (seed-URL neighbors dominate) for the link graph to be a useful signal. It is off by default and exposed as a tunable parameter.

---

## 19. Web Crawler

**File:** [`searchengine/crawler.py`](searchengine/crawler.py)

### Architecture

```
asyncio event loop
    │
    ├── N coroutines (concurrency=16 by default)
    │       each: pop URL from frontier → robots.txt → fetch → parse → push links
    │
    └── Frontier: host-partitioned deque
                  (one queue per host, round-robin over hosts)
```

### Key Design Decisions

**1. robots.txt compliance:**
```python
rp = urobot.RobotFileParser()
rp.parse(body.splitlines())
allowed = rp.can_fetch(UA, url)
```
Disrespecting robots.txt gets IP bans. The crawler honors `Crawl-Delay` directives.

**2. 429/503 exponential backoff:**
```python
if r.status in (429, 503):
    cur = self.host_delay.get(host, self.delay)
    self.host_delay[host] = min(60.0, max(cur * 2.0, wait, 2.0))
```
On rate-limiting (429 = Too Many Requests, 503 = Service Unavailable), the per-host delay doubles. On success, it decays by 10%.

**3. SHA-1 deduplication:**
```python
h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
if h in self.hashes:
    return False  # exact duplicate
```
Prevents storing mirror pages.

**4. Isolation principle:** Every `_step` call is wrapped in `try/except`. One broken page can never kill the entire crawl.

### Output Files

```
out_dir/
├── pages.tsv       # did\ttext (input to indexer)
├── meta.jsonl      # per-page: url, title, sha1, length
├── links.tsv       # from_url\tto_url (for PageRank)
└── crawl_stats.json
```

---

## 20. Distributed Search — Sharding & Replication

**Files:** [`searchengine/distributed/`](searchengine/distributed/)

### Document Partitioning

Each shard holds a **contiguous docid range**. Every shard sees every query term, and shards work in parallel:

```
collection.tsv (8.8M docs)
    │
    ├── shard 0: docs 0 – 2.2M   → shard server :8110
    ├── shard 1: docs 2.2M – 4.4M → shard server :8111
    ├── shard 2: docs 4.4M – 6.6M → shard server :8112
    └── shard 3: docs 6.6M – 8.8M → shard server :8113
                                          │
                                    broker :9000
                                    (scatter-gather)
```

### The BM25 Sharding Problem

IDF = $\log(1 + (N - \text{df} + 0.5) / (\text{df} + 0.5))$

$N$ and $\text{df}$ are **collection-wide** statistics. If shard 0 only knows its own 2.2M documents, it computes different IDF values, and the same document scores differently depending on how the corpus was split.

### Global Statistics Solution

```python
# shard_build.py: --global-stats flag
# 1. Build a full index over the whole corpus to get accurate df/N/avgdl
# 2. Share these statistics with every shard
# 3. Every shard rescores its postings using GLOBAL df

def _rescore_with_global(index_dir, gdf, n_docs, avgdl):
    # For each term, use gdf[term] (global df) instead of local df
    idf = np.log(1.0 + (n_docs - gdf_arr + 0.5) / (gdf_arr + 0.5))
    impacts = idf_p * tf_f * (k1 + 1.0) / (tf_f + k1 * (...))
```

**Measured impact:** Without global stats, sharding silently changes **8.7%** of top-10 results compared to the single-node index. With global stats, the results are identical.

### The Broker (Scatter-Gather)

```python
# broker.py: sends every query to all shards, merges results
# Request: HTTP GET to each shard server
# Response: JSON list of (local_docid, score)
# Merge: sort all responses by score, translate local → global docids, return top-k
```

For replicas: the broker maintains a list of addresses per shard and round-robins.

**Measured performance:** On one machine, 4-shard distributed is **4.8× slower** than single-node (network overhead for 4 HTTP round trips > the parallelism benefit). Distribution buys **capacity and availability**, not speed, when all shards run on the same machine.

---

## 21. Live Index — Segments, Deletes & Tiered Merges

**Files:** [`searchengine/live/writer.py`](searchengine/live/writer.py), [`searchengine/live/reader.py`](searchengine/live/reader.py)

### The Problem

A static index can't be updated without a full rebuild. The live index adds new documents without rebuilding the whole corpus.

### Segments

Each write creates a new **segment** — a small, complete index:

```
live_index/
├── manifest.json           # list of active segments + metadata
├── seg00001/idx/           # segment 1: small index (new documents)
├── seg00002/idx/           # segment 2: another batch
└── seg00003/idx/           # merged segment (larger, fewer)
```

### Write Path (writer.py)

```
new docs → tokenize → build small index → compress → write segment dir
→ atomically update manifest.json (rename trick for crash safety)
```

### Read Path (reader.py)

```python
def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
    merged = []
    for seg, s, deletes, ids in self.segs:
        want = k if deletes is None else min(k * 4, k + deletes.sum())
        for pid, score in s.search(query, want):
            if deletes is not None and deletes[pid]:
                continue             # skip tombstoned documents
            merged.append((int(ids[pid]), score))
    merged.sort(key=lambda h: -h[1])
    return merged[:k]
```

Each segment is queried independently. Results are merged by score. Local docids are translated to global docids via the `ids` mapping.

**Over-fetching for tombstones:** If a segment has deleted documents, we fetch `k + num_deletes` so deletions can't shrink the result below k.

### Tiered Merge Policy (Lucene-inspired)

```python
def _tiered_candidates(segs, max_segments, merge_factor):
    live = [s for s in segs if s.get("live", True)]
    if len(live) <= max_segments:
        return []           # already tidy
    live.sort(key=lambda s: s["n_docs"])
    return live[:merge_factor]   # merge the smallest segments
```

**Why tiered (not "merge everything")?** Merging the 4 smallest segments into 1 reduces segment count without touching large segments. A naive "merge everything on every write" would be **quadratic**: writing 1M documents one at a time would rewrite the entire index 1M times.

### Merge refreshes IDF

During a merge, impacts are recomputed using current collection statistics:

```python
# Gather df from all surviving segments
for s in manifest["segments"]:
    for t, c in load_df(s).items():
        gdf[t] = gdf.get(t, 0) + c
rescore_with_global(raw, gdf, gn, gtotal / gn)
```

This bounds the **IDF drift** that incremental writes introduce (adding new documents changes N and df, making old precomputed impacts stale).

---

## 22. HTTP Server & Search UI

**Files:** [`searchengine/server.py`](searchengine/server.py), [`searchengine/ui.py`](searchengine/ui.py)

### Multi-Process Architecture

```python
# server.py
import gc
gc.freeze()                      # freeze GC before fork (no GC pauses in children)
for _ in range(max(0, workers - 1)):
    if os.fork() == 0:
        break   # child worker
```

`os.fork()` creates a copy of the process. All workers share the same mmap'd index (OS page cache) with **zero copying cost** — the kernel maps the same physical pages into all processes. This is how Apache / Nginx / Gunicorn achieve multi-process serving with shared memory.

### Why Fork, Not Threads?

- Python's GIL serializes threads for CPU-bound work
- The mmap'd arrays are truly shared between fork'd processes via page cache
- The ONNX encoder is built **after** fork to avoid forking a multi-threaded process (known to deadlock)

### API Endpoints

```
GET /search?q=<query>&k=10&rerank=1
    → JSON: {hits, took_ms, ranking, did_you_mean, phrase}

GET /suggest?q=<prefix>&k=6
    → JSON: {suggestions}
```

### Search UI (ui.py)

The UI is a single self-contained HTML string with:
- **Autocomplete:** 90ms debounce on `input` → fetch `/suggest` → dropdown
- **Arrow key navigation** through suggestions
- **"Did you mean"** link → click to re-search with corrected query
- **Highlighted snippets:** JavaScript regex replaces query terms with `<mark>` tags
- **Mode selector:** `auto` (hybrid if available), `bm25 only`, `precision (slow)` (cross-encoder)

---

## 23. Forex Application Layer

**Files:** [`forex_app/`](forex_app/)

This is a domain-specific application built on top of the search engine core.

### Architecture

```
FastAPI (forex_app/web/app.py)
    │
    ├── BM25 Search (forex_app/engine/search.py)
    │       multi-field boosting: pairs ×3.0, title ×2.5, summary ×1.2
    │
    ├── News Fetcher (forex_app/fetcher/news_service.py)
    │       RSS feeds: FXStreet, Yahoo Finance, CNBC, MarketWatch, Fed, BoE
    │       background auto-refresh every N minutes
    │
    ├── NLP Classifier (forex_app/nlp/classifier.py)
    │       regex-based currency pair detection
    │       sentiment: positive/negative/neutral keyword scoring
    │       impact level: high/medium/low
    │
    └── Storage Layer
            ├── SQLite (forex_app/storage/database.py)
            │       persistent, ACID, indexed by pair/date/sentiment
            └── Cache (forex_app/storage/cache.py)
                    dual-layer: Python dict (in-memory) + optional Redis
```

### Multi-Field BM25 (forex_app/engine/search.py)

```python
# Field boost weights
FIELD_WEIGHTS = {
    "pair": 3.0,    # currency pair match is most valuable
    "title": 2.5,   # headline keyword match
    "summary": 1.2, # body text match
}

# The score combines BM25 over all fields
total_score = sum(bm25_score(field, query) * weight
                  for field, weight in FIELD_WEIGHTS.items())
```

### Dual-Layer Cache (forex_app/storage/cache.py)

```
Client Request
    │
    ▼
In-Memory Cache (Python dict + TTL)
    │ HIT: return instantly (<0.1ms)
    │ MISS ↓
    ▼
Redis Cache (if available on port 6379)
    │ HIT: return, populate in-memory
    │ MISS ↓
    ▼
SQLite Database
    │ return, populate both cache layers
```

**TTL (Time-to-Live):** Cached entries expire after N seconds to ensure freshness of forex news.

### RSS Feed Parser (forex_app/fetcher/rss_feeds.py)

```python
# Handles both RSS 2.0 and Atom 1.0 feed formats
# XML sanitization: strips namespace prefixes, handles malformed XML
# Deduplication: title+URL hash prevents duplicate articles
```

### Currency Pair Detection (forex_app/nlp/classifier.py)

```python
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
         "USD/CHF", "NZD/USD", "XAU/USD", "DXY", ...]

# Multi-pattern regex per pair + currency name variants
# e.g. "Euro Dollar" → EUR/USD
```

---

## 24. Benchmarking & Quality Measurement

**Files:** [`bench/`](bench/)

### Quality Metric: MRR@10

**Mean Reciprocal Rank** measures how high in the result list the first relevant document appears:

$$\text{MRR@10} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the position (1-indexed) of the first relevant result for query $i$, capped at 10 (if no relevant result in top-10, contribution is 0).

**Example:**
```
Query 1: relevant doc at position 1 → 1/1 = 1.0
Query 2: relevant doc at position 3 → 1/3 = 0.333
Query 3: no relevant in top-10     → 0
MRR@10 = (1.0 + 0.333 + 0) / 3 = 0.444
```

### BEIR Benchmark — Out-of-Domain Generalization

**BEIR** (Benchmarking IR) tests how well a model trained on one corpus generalizes to others. This engine measures nDCG@10 on:

| Dataset | Domain | BM25 nDCG@10 | Hybrid nDCG@10 |
|---------|--------|-------------|----------------|
| ArguAna | Argument retrieval | — | — |
| FiQA | Financial QA | — | — |
| NFCorpus | Medical | — | — |
| SciFact | Scientific fact checking | 0.688 | 0.708 (α=0.5) |

### Benchmark Files

| File | What it measures |
|------|-----------------|
| `bench/build/bench_build.py` | Index build time vs corpus size |
| `bench/hybrid/bench_hybrid.py` | Hybrid search latency & MRR |
| `bench/hybrid/bench_encoder.py` | Encoder throughput (passages/sec) |
| `bench/distributed/bench_dist.py` | Distributed latency & QPS |
| `bench/serving/loadgen.py` | Load testing with concurrent requests |
| `bench/beir/run_beir.py` | Out-of-domain BEIR evaluation |
| `bench/quality/tune_k1b.py` | Grid search for optimal k1, b |

### Measured Performance (Sealed Dev Set)

```json
{
  "mrr_at_10_full": 0.18913,
  "cold_p50_ms": 1.47,
  "cold_p99_ms": 18.45,
  "warm_p50_ms": 1.46,
  "warm_p99_ms": 18.43,
  "warm_qps_1core": 362.8
}
```

"Cold" = first query (arrays not in OS page cache). "Warm" = cached. The near-identical results show the index fits in the OS file cache after the first query.

---

## 25. Libraries & Tools Reference

### Python Standard Library

| Module | Used for |
|--------|---------|
| `re` | Tokenizer regex, phrase search regex |
| `heapq` | Top-k heap in v0 search |
| `math` | `math.log` for BM25 IDF |
| `pickle` | Vocabulary serialization (`terms.pkl`) |
| `json` | Index metadata (`meta.json`), API responses |
| `ctypes` | Python ↔ C bridge (MaxScore, BMW kernels) |
| `threading` | Per-thread scratch buffers for C kernel |
| `subprocess` | Launch native scanner processes, compile C |
| `bisect` | Binary search for autocomplete |
| `hashlib` | SHA-1 deduplication in crawler |
| `asyncio` | Async/await for concurrent crawling |
| `os.fork` | Multi-process server |
| `http.server` | `ThreadingHTTPServer` for shard servers |

### NumPy

| Operation | Why used |
|-----------|---------|
| `np.load(..., mmap_mode="r")` | Memory-map arrays — OS manages page cache, no RAM copy |
| `scores[docids[s:e]] += impacts[s:e]` | Vectorized fancy-index accumulation |
| `np.argpartition(arr, -k)` | O(n) top-k selection (quickselect) |
| `np.searchsorted` | Binary search for intersection |
| `np.maximum.reduceat` | Per-term max impact (segmented reduction) |
| `np.repeat` | Expand per-term arrays to per-posting |
| `np.add.at` | Unbuffered add (no += de-duplication) |
| `np.memmap` | `blob.bin` for compressed index |

### ONNX Runtime (`onnxruntime`)

| Feature | Purpose |
|---------|---------|
| `ort.InferenceSession` | Load and run ONNX model |
| `SessionOptions.intra_op_num_threads` | Control BLAS parallelism |
| `ORT_ENABLE_ALL` | Maximum graph fusion |
| `CPUExecutionProvider` | CPU-only (CoreML EP OOM-killed the machine) |
| `quantize_dynamic` | int8 weight quantization |

### HuggingFace `tokenizers`

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("models/minilm/tokenizer.json")
tok.enable_truncation(max_length=256)
tok.no_padding()                       # padding done manually in _forward
id_lists = [e.ids for e in tok.encode_batch(texts)]
```

Fast Rust-based tokenizer — 10–100× faster than Python tokenizers.

### `aiohttp` (Crawler)

Async HTTP client with:
- Connection pooling (`TCPConnector(limit=concurrency)`)
- DNS caching (`ttl_dns_cache=300`)
- Redirect following (`allow_redirects=True, max_redirects=5`)
- Timeout handling (`ClientTimeout(total=TIMEOUT)`)

### FastAPI (Forex App)

```python
# Async REST endpoints
@app.get("/api/news")
@app.get("/api/search")
@app.get("/api/stream")   # Server-Sent Events for live updates
```

### SQLite (`sqlite3`)

Used for persistent article storage in the forex app:
- ACID transactions
- Indexed by `(currency_pair, published_date, sentiment)`
- Deduplication by `(title, url)` hash

---

## 26. Parameters Encyclopedia

Every tunable parameter, what it does, and its measured value.

### BM25 Parameters

| Parameter | Default | Tuned | File | Effect |
|-----------|---------|-------|------|--------|
| `k1` | 0.9 | **0.82** | `indexer_v4.py`, `search.py` | TF saturation. Lower = faster saturation. Tuned on MS MARCO dev MRR. |
| `b` | 0.4 | **0.75** | `indexer_v4.py`, `search.py` | Length normalization. Higher = stronger normalization. MS MARCO passages are short → stronger b helps. |
| `K1` (search.py constant) | 0.9 | — | `search.py` | v0 fallback, untuned |
| `B` (search.py constant) | 0.4 | — | `search.py` | v0 fallback, untuned |

### Hybrid Parameters

| Parameter | Value | File | Reason |
|-----------|-------|------|--------|
| `ALPHA_IN_DOMAIN` | 0.1 | `search_hybrid.py` | MS MARCO best (experiments/tune_alpha.py) |
| `ALPHA_UNKNOWN_DOMAIN` | 0.5 | `search_hybrid.py` | Safe starting point for unknown corpora |
| `TOPK_RERANK` | 1000 | `search_hybrid.py` | Recall@1000 = 0.87, p99 = 47ms |

### Encoder Parameters

| Parameter | Value | File | Reason |
|-----------|-------|------|--------|
| `MODEL_DIR` | `models/minilm` | `encoder.py` | MiniLM-L6-v2: 384-dim, fast |
| `max_len` | 256 | `encoder.py` | MS MARCO passages ≤ 256 subwords |
| `dim` | 384 | `encoder.py` | MiniLM hidden size |
| `batch` | 64 | `encoder.py` | GPU batch size for embedding |
| `threads` | 6 | `encoder.py` | ONNX Runtime intra-op threads |
| `quantized` | True | `encoder.py` | int8 weights, 2-3× speedup |

### PQ Parameters

| Parameter | Value | File | Reason |
|-----------|-------|------|--------|
| `M` | 64 | `pq.py` | 64 subspaces × 6 dims. 24× compression |
| `M` (DenseIndex PQ) | 96 | `dense.py` | 96 subspaces for web index (smaller corpus) |
| `256` centroids | — | `pq.py` | 1 byte per subspace (8-bit codes) |
| `iters` | 20 | `pq.py` | k-means iterations. Convergence usually by 15 |
| `EXACT_MAX_DOCS` | 1,000,000 | `dense.py` | Above this, use PQ. Below, keep exact f32 |

### Compression Parameters

| Parameter | Value | File | Reason |
|-----------|-------|------|--------|
| `TF_CLAMP` | 63 | `indexer_v4.py` | BM25 saturates; TF > 63 adds ~nothing. Fits in 6 bits |
| Block size | 128 | `compress_index.py` | PISA standard. Cache-line aligned. |
| `quant_scale` | `max(impact)/255` | `compress_index.py` | Global scale for distributed consistency |

### Spell Correction Parameters

| Parameter | Value | File | Reason |
|-----------|-------|------|--------|
| `RARE` | 5 | `spell.py` | Terms in < 5 docs are suspicious |
| d1 threshold | `wdf × 100` | `spell.py` | Candidate must be 100× more common |
| d2 threshold | `wdf × 1000` | `spell.py` | Candidate must be 1000× more common |
| Alphabet | `a-z0-9` | `spell.py` | Matches tokenizer's character set |

### Cross-Encoder Parameters

| Parameter | Value | File | Reason |
|-----------|-------|------|--------|
| `depth` | 20 | `search_hybrid.py` | Optimal shortlist (measured on TRAIN, not dev) |
| `threads` | 4 | `cross_encoder.py` | ONNX thread count for CE |

### IVF Parameters

| Parameter | Value | File | Reason |
|-----------|-------|------|--------|
| `n_clusters` | 4096 | `ann.py` | ~2155 docs/cluster at 8.84M |
| `nprobe` | 32 | `ann.py` | Visit 32/4096 ≈ 0.8% of clusters |
| `sample` | 400,000 | `ann.py` | Train k-means on 400k docs |
| `chunk` | 200,000 | `ann.py` | Assignment chunk to control peak RAM |

### Live Index Parameters

| Parameter | Value | File | Reason |
|-----------|-------|------|--------|
| `max_segments` | 8 | `live/reader.py` | Merge when > 8 segments |
| `merge_factor` | 4 | `live/reader.py` | Merge the 4 smallest at once |

### Server Parameters

| Parameter | Default | File | |
|-----------|---------|------|--|
| `workers` | 4 | `server.py` | Fork-based worker processes |
| `port` | 8000 | `server.py` | HTTP server port |
| `ENC_THREADS` | 2 | `server.py` | Encoder threads per worker |

### Crawler Parameters

| Parameter | Default | File | |
|-----------|---------|------|--|
| `max_pages` | 5000 | `crawler.py` | Crawl budget |
| `concurrency` | 16 | `crawler.py` | Async HTTP workers |
| `delay` | 1.0s | `crawler.py` | Per-host politeness delay |
| `TIMEOUT` | 15s | `crawler.py` | Per-request timeout |
| `MAX_BYTES` | 1MB | `crawler.py` | Max HTML body to read |

---

## 27. Data Flow: End-to-End Query Lifecycle

```
User types: "What is the federal reserve interest rate?"
                        │
                        ▼
        ┌───────────────────────────────┐
        │         JavaScript UI         │
        │  90ms debounce → /suggest     │
        │  on Enter/click → /search     │
        └───────────────────────────────┘
                        │  GET /search?q=...&k=10&rerank=1
                        ▼
        ┌───────────────────────────────┐
        │    server.py Handler          │
        │  parse query params           │
        │  check spell correction       │
        │  detect quoted phrases        │
        └───────────────────────────────┘
                        │
           ┌────────────┴────────────┐
           │  phrase?    no phrase   │
           ▼                         ▼
    PhraseSearcher            HybridSearcher
    parse_query()             search(query, k=10)
    ranked-walk                      │
    verify regex                     ├─ bm25.search(query, 1000)
                                     │    tokenize → stem
                                     │    term IDs → offset slices
                                     │    C MaxScore kernel
                                     │    → top-1000 (docid, bm25_score)
                                     │
                                     ├─ enc.encode([query])
                                     │    ONNX Runtime (MiniLM)
                                     │    mean pooling → L2 normalize
                                     │    → float32[384]
                                     │
                                     ├─ PQ.adc(lut(qv), codes[pids])
                                     │    64 table lookups per candidate
                                     │    → dense score per candidate
                                     │
                                     └─ blend = 0.1*minmax(bm25) + 0.9*minmax(dense)
                                          argsort → top-10
                        │
                        ▼
        ┌───────────────────────────────┐
        │    Fetch snippets from store  │
        │    Build JSON response        │
        │    {hits, took_ms, ranking}   │
        └───────────────────────────────┘
                        │
                        ▼
        JavaScript highlights query terms in snippets
        Renders results with score, doc ID, URL, snippet
```

---

## 28. Index Files Reference

### V1 (Flat) Index

| File | Type | Shape | Content |
|------|------|-------|---------|
| `meta.json` | JSON | — | `n_docs, n_terms, avgdl, k1, b, stemmed` |
| `terms.pkl` | pickle | dict | `{stem: term_id}` |
| `offsets.u64.npy` | uint64 | `[n_terms+1]` | CSR row pointers |
| `docids.u32.npy` | uint32 | `[total_postings]` | Document IDs per posting |
| `impacts.f32.npy` | float32 | `[total_postings]` | Precomputed BM25 scores |
| `tfs.u8.npy` | uint8 | `[total_postings]` | Raw term frequencies |
| `doclens.u32.npy` | uint32 | `[n_docs]` | Token count per document |
| `max_impact.f32.npy` | float32 | `[n_terms]` | Max impact per term (MaxScore) |

### V3 (Compressed) Index

| File | Type | Shape | Content |
|------|------|-------|---------|
| `meta.json` | JSON | — | + `format: "c1"`, `quant_scale`, `n_blocks` |
| `terms.pkl` | pickle | dict | Same as v1 |
| `blob.bin` | uint8 | `[blob_bytes]` | Bitpacked deltas + u8 impacts |
| `block_last.u32.npy` | uint32 | `[n_blocks]` | Last docid per block |
| `block_width.u8.npy` | uint8 | `[n_blocks]` | Bit width per block |
| `block_maxq.u8.npy` | uint8 | `[n_blocks]` | Max quantized impact per block |
| `term_block_start.i64.npy` | int64 | `[n_terms+1]` | Block CSR pointers |
| `dfs.i64.npy` | int64 | `[n_terms]` | Document frequencies |
| `doclens.u32.npy` | uint32 | `[n_docs]` | Same as v1 |

### Dense Index

| File | Type | Content |
|------|------|---------|
| `meta.json` | JSON | `mode, model_dir, dim, n_docs` |
| `vectors.f32.npy` | float32 | Exact embeddings (mode="exact") |
| `pq_centroids.npy` | float32 | `(M, 256, dsub)` centroid array |
| `codes.u8.npy` | uint8 | `(n_docs, M)` PQ codes |

### IVF Index

| File | Type | Content |
|------|------|---------|
| `meta.json` | JSON | `n_docs, n_clusters, dense_dir` |
| `centroids.f32.npy` | float32 | `(n_clusters, dim)` coarse centroids |
| `postings.i32.npy` | int32 | Docids sorted by cluster |
| `offsets.i64.npy` | int64 | `(n_clusters+1,)` CSR offsets |

---

## Quick Start (for new learners)

### Step 1: Build a tiny index

```bash
# The demo collection is already in the repo (6 documents)
python3 -c "
from searchengine.indexer_v4 import build
import json
result = build('data/demo/collection.tsv', 'indexes/my_first', workers=1)
print(json.dumps(result, indent=2))
"
```

### Step 2: Search it

```bash
python3 -c "
from searchengine.search_v1 import SearcherV1
s = SearcherV1('indexes/my_first')
hits = s.search('interest rate federal reserve', k=5)
for docid, score in hits:
    print(f'  doc {docid}  score {score:.4f}')
"
```

### Step 3: Try BM25 math manually

```python
import math

# Corpus: 3 documents
docs = [
    "the cat sat on the mat",
    "the dog ran on the grass",
    "the cat ran fast",
]

# Parameters
k1, b = 0.82, 0.75
query = ["cat", "ran"]

# Build index
from collections import defaultdict
postings = defaultdict(list)
doc_lens = []
for i, d in enumerate(docs):
    toks = d.split()
    tf = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1
    for t, f in tf.items():
        postings[t].append((i, f))
    doc_lens.append(len(toks))

n_docs = len(docs)
avgdl = sum(doc_lens) / n_docs
print(f"avgdl = {avgdl:.2f}")

scores = {}
for t in query:
    if t not in postings:
        continue
    df = len(postings[t])
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    print(f"\nterm='{t}', df={df}, idf={idf:.4f}")
    for docid, tf in postings[t]:
        dl = doc_lens[docid]
        s = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        scores[docid] = scores.get(docid, 0) + s
        print(f"  doc {docid}: tf={tf}, dl={dl}, score_contribution={s:.4f}")

print("\nFinal scores:")
for docid, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  doc {docid} '{docs[docid]}': {score:.4f}")
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Inverted index** | Maps terms → list of documents containing them |
| **Posting list** | The list of (docid, tf) pairs for a single term |
| **TF** | Term frequency — how many times a term appears in a document |
| **IDF** | Inverse document frequency — how rare a term is across the corpus |
| **BM25** | Best Match 25 — the standard probabilistic ranking model |
| **Impact** | Precomputed BM25 contribution of a single posting |
| **MaxScore** | Algorithm to skip non-competitive documents without evaluating them |
| **WAND** | Weak AND — another top-k algorithm; BMW extends it with block skips |
| **Bi-encoder** | Encodes query and document independently → dot product for similarity |
| **Cross-encoder** | Jointly encodes query + document → direct relevance score |
| **PQ** | Product quantization — vector compression via subspace clustering |
| **ADC** | Asymmetric distance computation — full-precision query, quantized docs |
| **IVF** | Inverted file — coarse quantization for ANN search |
| **MRR@10** | Mean Reciprocal Rank at cutoff 10 — quality metric |
| **nDCG@10** | Normalized Discounted Cumulative Gain — quality metric (graded relevance) |
| **CSR** | Compressed Sparse Row — the layout for storing posting lists as flat arrays |
| **Delta coding** | Store differences between consecutive docids (smaller values = fewer bits) |
| **Bit packing** | Pack multiple small integers into a byte stream at the minimum required width |
| **mmap** | Memory-mapped file — OS manages paging between disk and RAM |
| **Segment** | A small, complete index created by one batch of writes (live index) |
| **Tiered merge** | Merge similarly-sized segments to avoid quadratic rebuild cost |
| **PageRank** | Link-based page importance score (damping factor, power iteration) |
| **Porter stemmer** | Rule-based algorithm reducing inflected words to their stem |
| **Lemmatization** | Vocabulary-based reduction to dictionary base form (not used here) |
| **BEIR** | Benchmarking IR — a suite of out-of-domain retrieval datasets |
| **MS MARCO** | Microsoft MAchine Reading COmprehension — 8.84M passage corpus used for training |
| **Alpha (α)** | BM25 weight in hybrid fusion (1-α = dense weight) |

---

*This document covers every module in the `searchengine/` and `forex_app/` packages. Reading order: sections 1 → 4 → 5 → 6 → 12 gives you 80% of the value in ~45 minutes. The rest are deep-dives for specific subsystems.*
