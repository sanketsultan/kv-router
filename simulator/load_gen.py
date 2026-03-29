"""
Load generator — demonstrates the cache hit improvement in two modes:

  Mode A (naive):    round-robin across backends — simulates what happens
                     WITHOUT a cache-aware router.

  Mode B (smart):    send all requests through the KV-cache-aware router —
                     demonstrates the TTFT improvement.

At the end prints a side-by-side comparison showing TTFT p50/p95 and
cache hit rate for both strategies.

Usage:
  # Run all backends + router first, then:
  python load_gen.py --mode router   # test the smart router
  python load_gen.py --mode naive    # test dumb round-robin
  python load_gen.py --mode both     # compare both (default)
"""

import argparse
import json
import random
import statistics
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

import os as _os
_router_base = _os.getenv("ROUTER_URL", "http://localhost:8000")
ROUTER_URL   = f"{_router_base}/v1/chat/completions"
BACKEND_URLS = [
    "http://localhost:8001/v1/chat/completions",
    "http://localhost:8002/v1/chat/completions",
    "http://localhost:8003/v1/chat/completions",
]

NUM_REQUESTS        = 60
CONCURRENCY         = 5
SHARED_PROMPT_RATIO = 0.70   # 70% of requests share the same system prompt
                               # (these should benefit from cache hits)

# Shared system prompt — long enough to be expensive to prefill
SHARED_SYSTEM_PROMPT = """You are an expert AI assistant for a financial services company.
You help analysts understand complex regulatory documents, financial statements, and
market data. You always cite your sources, explain your reasoning step by step, and
flag any uncertainties. You are familiar with IFRS, GAAP, Basel III, MiFID II, and
Dodd-Frank regulations. When asked about financial instruments, you provide both
technical definitions and practical implications for portfolio managers."""

# Unique system prompts (simulate different customers / departments)
UNIQUE_SYSTEM_PROMPTS = [
    "You are a helpful coding assistant specializing in Python and distributed systems.",
    "You are a medical information assistant. Always recommend consulting a doctor.",
    "You are a customer support agent for a SaaS product. Be concise and friendly.",
    "You are a legal research assistant. Cite relevant case law where applicable.",
    "You are a data science tutor helping beginners learn machine learning concepts.",
]

USER_MESSAGES = [
    "Explain the key differences between IFRS 9 and GAAP ASC 326 for loan loss provisioning.",
    "What are the Basel III capital adequacy requirements for tier-1 capital?",
    "Summarize the main provisions of MiFID II regarding algorithmic trading.",
    "How does the Dodd-Frank Act affect over-the-counter derivatives trading?",
    "What is the difference between market risk and credit risk in a trading book?",
    "Explain collateral management best practices for repo agreements.",
    "What metrics should I use to evaluate a hedge fund's performance?",
    "How do I interpret a company's debt-to-equity ratio in context?",
]


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

@dataclass
class Result:
    target: str        # "router" or "naive"
    latency_ms: float
    cache_header: str  # value of X-Cache response header (or "n/a")
    success: bool
    error: str = ""


def _send(url: str, messages: list[dict], timeout: int = 10) -> tuple[float, str, bool, str]:
    payload = json.dumps({"model": "fake-llm-70b", "messages": messages}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            cache_hdr = resp.headers.get("X-Cache", "n/a")
            return latency_ms, cache_hdr, True, ""
    except urllib.error.URLError as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return latency_ms, "n/a", False, str(e)


def _build_messages(shared: bool) -> list[dict]:
    if shared:
        system = SHARED_SYSTEM_PROMPT
    else:
        system = random.choice(UNIQUE_SYSTEM_PROMPTS)
    user = random.choice(USER_MESSAGES)
    return [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
    ]


# ---------------------------------------------------------------------------
# Test runs
# ---------------------------------------------------------------------------

def run_naive(n: int, concurrency: int) -> list[Result]:
    """Round-robin across backends, no cache awareness."""
    results = []
    rr_index = [0]

    def task(i: int) -> Result:
        url = BACKEND_URLS[rr_index[0] % len(BACKEND_URLS)]
        rr_index[0] += 1
        shared = random.random() < SHARED_PROMPT_RATIO
        messages = _build_messages(shared)
        lat, cache, ok, err = _send(url, messages)
        return Result("naive", lat, cache, ok, err)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(task, i) for i in range(n)]
        for f in as_completed(futures):
            results.append(f.result())
            _progress(len(results), n, "naive")

    print()
    return results


def run_router(n: int, concurrency: int) -> list[Result]:
    """Send through the KV-cache-aware router."""
    results = []

    def task(i: int) -> Result:
        shared = random.random() < SHARED_PROMPT_RATIO
        messages = _build_messages(shared)
        lat, cache, ok, err = _send(ROUTER_URL, messages)
        return Result("router", lat, cache, ok, err)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(task, i) for i in range(n)]
        for f in as_completed(futures):
            results.append(f.result())
            _progress(len(results), n, "router")

    print()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _progress(done: int, total: int, label: str):
    bar = "#" * int(done / total * 30)
    print(f"\r  [{label:6s}] [{bar:<30s}] {done}/{total}", end="", flush=True)


def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def print_report(naive: list[Result], router: list[Result]):
    print("\n" + "=" * 62)
    print("  KV-Cache-Aware Router — Benchmark Results")
    print("=" * 62)

    for label, results in [("Naive (round-robin)", naive), ("Smart (cache-aware)", router)]:
        ok = [r for r in results if r.success]
        latencies = [r.latency_ms for r in ok]
        hits = sum(1 for r in ok if r.cache_header == "hit")
        hit_rate = hits / len(ok) if ok else 0

        print(f"\n  {label}")
        print(f"  {'─' * 40}")
        print(f"  Requests:       {len(ok)}/{len(results)} succeeded")
        print(f"  Cache hit rate: {hit_rate:.0%}  ({hits}/{len(ok)} hits)")
        if latencies:
            print(f"  TTFT p50:       {_percentile(latencies, 50):.0f}ms")
            print(f"  TTFT p95:       {_percentile(latencies, 95):.0f}ms")
            print(f"  TTFT mean:      {statistics.mean(latencies):.0f}ms")

    # Improvement summary
    if naive and router:
        naive_ok   = [r for r in naive   if r.success]
        router_ok  = [r for r in router  if r.success]
        if naive_ok and router_ok:
            naive_p50  = _percentile([r.latency_ms for r in naive_ok],  50)
            router_p50 = _percentile([r.latency_ms for r in router_ok], 50)
            improvement = (naive_p50 - router_p50) / naive_p50 * 100 if naive_p50 else 0

            router_hits = sum(1 for r in router_ok if r.cache_header == "hit")
            router_hit_rate = router_hits / len(router_ok)

            print(f"\n  {'─' * 40}")
            print(f"  TTFT improvement (p50):  {improvement:.1f}%")
            print(f"  Router cache hit rate:   {router_hit_rate:.0%}")
            print(
                f"\n  Interpretation: The cache-aware router reduced median TTFT by "
                f"{improvement:.0f}% by routing {router_hit_rate:.0%} of requests to a "
                f"replica that already had the KV blocks computed."
            )

    print("\n" + "=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Router load generator")
    parser.add_argument(
        "--mode", choices=["router", "naive", "both"], default="both",
        help="Which routing strategy to test"
    )
    parser.add_argument("--requests", type=int, default=NUM_REQUESTS)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    naive_results  = []
    router_results = []

    if args.mode in ("naive", "both"):
        print(f"\nRunning NAIVE (round-robin) — {args.requests} requests @ concurrency={args.concurrency}")
        naive_results = run_naive(args.requests, args.concurrency)

    if args.mode in ("router", "both"):
        print(f"\nRunning SMART (cache-aware router) — {args.requests} requests @ concurrency={args.concurrency}")
        router_results = run_router(args.requests, args.concurrency)

    if args.mode == "router":
        naive_results = router_results  # avoid empty comparison
        print_report([], router_results)
    elif args.mode == "naive":
        print_report(naive_results, [])
    else:
        print_report(naive_results, router_results)
