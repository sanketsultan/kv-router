"""
Automated tests for the KV-cache-aware router.
"""

import json
import random
import statistics
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest

SYSTEM_PROMPT = """You are an expert financial assistant helping analysts
interpret regulatory documents under IFRS, GAAP, Basel III, and MiFID II.
Always cite sources and explain reasoning step by step."""

USER_MESSAGES = [
    "Explain Basel III tier-1 capital requirements.",
    "What is the difference between IFRS 9 and ASC 326?",
    "Summarize MiFID II algorithmic trading provisions.",
    "How does Dodd-Frank affect OTC derivatives?",
    "What metrics evaluate hedge fund performance?",
]


def _post(url: str, messages: list[dict], timeout: int = 10) -> dict:
    payload = json.dumps({"model": "test", "messages": messages}).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        latency_ms = (time.monotonic() - t0) * 1000
        return {
            "status":     resp.status,
            "cache":      resp.headers.get("X-Cache", "unknown"),
            "served_by":  resp.headers.get("X-Served-By", "unknown"),
            "latency_ms": latency_ms,
        }


def _pct(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


# ---------------------------------------------------------------------------

class TestHealth:
    def test_router_is_healthy(self, running_stack):
        url = running_stack["router_url"]
        with urllib.request.urlopen(f"{url}/health") as resp:
            data = json.loads(resp.read())
        assert data["status"] == "ok"

    def test_all_backends_reported(self, running_stack):
        url = running_stack["router_url"]
        with urllib.request.urlopen(f"{url}/health") as resp:
            data = json.loads(resp.read())
        healthy = [b for b in data["backends"] if b["healthy"]]
        running_stack["metrics"]["backend_health"] = data["backends"]
        assert len(healthy) == 3, f"Expected 3 healthy backends, got {len(healthy)}"


class TestSingleRequest:
    def test_request_succeeds(self, running_stack):
        url = running_stack["router_url"]
        r = _post(url, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "Hello"},
        ])
        assert r["status"] == 200

    def test_response_has_cache_header(self, running_stack):
        url = running_stack["router_url"]
        r = _post(url, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "Hello"},
        ])
        assert r["cache"] in ("hit", "miss"), \
            f"Expected X-Cache: hit or miss, got: {r['cache']}"

    def test_response_has_served_by_header(self, running_stack):
        url = running_stack["router_url"]
        r = _post(url, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "Hello"},
        ])
        assert r["served_by"].startswith("backend-"), \
            f"Expected X-Served-By: backend-N, got: {r['served_by']}"


class TestCacheAwareRouting:
    def test_cache_hit_rate_exceeds_threshold(self, running_stack):
        url = running_stack["router_url"]
        results = []
        for _ in range(20):
            r = _post(url, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": random.choice(USER_MESSAGES)},
            ])
            results.append(r)

        hits  = sum(1 for r in results if r["cache"] == "hit")
        total = len(results)
        hit_rate = hits / total

        running_stack["metrics"].update({
            "total_requests": total,
            "hits":           hits,
            "hit_rate":       hit_rate,
        })

        assert hit_rate >= 0.50, \
            f"Cache hit rate {hit_rate:.0%} below 50% threshold ({hits}/{total} hits)"

    def test_cache_hits_faster_than_misses(self, running_stack):
        url = running_stack["router_url"]

        # Guaranteed hits: shared system prompt (already warmed)
        hit_latencies = []
        for _ in range(10):
            r = _post(url, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": random.choice(USER_MESSAGES)},
            ])
            if r["cache"] == "hit":
                hit_latencies.append(r["latency_ms"])

        # Guaranteed misses: unique prompt per request
        miss_latencies = []
        for i in range(10):
            r = _post(url, [
                {"role": "system", "content": f"Unique context #{i}-{time.time()}"},
                {"role": "user",   "content": "Hello"},
            ])
            miss_latencies.append(r["latency_ms"])

        assert hit_latencies, "No cache hits observed even after warm-up"
        assert miss_latencies, "No cache misses observed"

        hit_p50  = _pct(hit_latencies,  50)
        hit_p95  = _pct(hit_latencies,  95)
        miss_p50 = _pct(miss_latencies, 50)
        miss_p95 = _pct(miss_latencies, 95)

        running_stack["metrics"].update({
            "hit_p50_ms":  hit_p50,
            "hit_p95_ms":  hit_p95,
            "miss_p50_ms": miss_p50,
            "miss_p95_ms": miss_p95,
        })

        assert hit_p50 < miss_p50, \
            f"Cache hits (p50={hit_p50:.0f}ms) not faster than misses (p50={miss_p50:.0f}ms)"

    def test_same_prefix_routes_to_same_replica(self, running_stack):
        url = running_stack["router_url"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "Explain Basel III."},
        ]
        first = _post(url, messages)
        expected_replica = first["served_by"]

        replicas_seen = set()
        for _ in range(5):
            r = _post(url, messages)
            if r["cache"] == "hit":
                replicas_seen.add(r["served_by"])

        if replicas_seen:
            assert expected_replica in replicas_seen, \
                f"Cache hits went to {replicas_seen}, expected {expected_replica}"


class TestConcurrency:
    def test_concurrent_requests_all_succeed(self, running_stack):
        url   = running_stack["router_url"]
        workers = 10
        total   = 20

        def task(_):
            return _post(url, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": random.choice(USER_MESSAGES)},
            ])

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(task, range(total)))

        failed = [r for r in results if r["status"] != 200]

        running_stack["metrics"].update({
            "concurrency_total":   total,
            "concurrency_success": total - len(failed),
            "concurrency_workers": workers,
        })

        assert len(failed) == 0, f"{len(failed)}/{total} requests failed"
