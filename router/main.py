"""
KV-Cache-Aware LLM Router

An OpenAI-compatible HTTP proxy that routes /v1/chat/completions requests
to the replica most likely to have the relevant KV cache already computed.

Environment variables:
  BACKENDS   — comma-separated list of backend URLs
               e.g. "http://backend-0:8001,http://backend-1:8002"
  PORT       — port to listen on (default 8000)
  LOG_LEVEL  — DEBUG / INFO / WARNING (default INFO)
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiohttp
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

from backend_pool import BackendPool
from prefix_tracker import PrefixTracker
from routing import select_replica

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("kv-router")

BACKEND_URLS = [
    url.strip()
    for url in os.getenv(
        "BACKENDS",
        "http://localhost:8001,http://localhost:8002,http://localhost:8003",
    ).split(",")
    if url.strip()
]

REPLICAS_CONFIG = [
    {"id": f"backend-{i}", "url": url}
    for i, url in enumerate(BACKEND_URLS)
]

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

requests_total = Counter(
    "kv_router_requests_total",
    "Total requests routed",
    ["replica", "cache_result"],   # cache_result: hit | miss
)
ttft_histogram = Histogram(
    "kv_router_ttft_milliseconds",
    "Time to first token (ms)",
    ["replica", "cache_result"],
    buckets=[50, 100, 200, 400, 600, 800, 1200, 2000, 4000],
)
in_flight_gauge = Gauge(
    "kv_router_in_flight_requests",
    "Current in-flight requests per replica",
    ["replica"],
)
cache_hit_rate_gauge = Gauge(
    "kv_router_cache_hit_rate",
    "Rolling cache hit rate per replica",
    ["replica"],
)
cached_prefixes_gauge = Gauge(
    "kv_router_cached_prefixes",
    "Number of prefix hashes tracked per replica",
    ["replica"],
)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

pool    = BackendPool(REPLICAS_CONFIG)
tracker = PrefixTracker()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await pool.start()
    asyncio.create_task(_metrics_updater())
    logger.info(
        "KV-cache-aware router started — %d backends: %s",
        len(REPLICAS_CONFIG),
        [r["url"] for r in REPLICAS_CONFIG],
    )
    yield
    await pool.stop()


app = FastAPI(title="KV-Cache-Aware LLM Router", version="0.1.0", lifespan=lifespan)


async def _metrics_updater() -> None:
    """Sync pool stats into Prometheus gauges every 5 s."""
    while True:
        for r in pool.snapshot():
            rid = r["id"]
            in_flight_gauge.labels(replica=rid).set(r["in_flight"])
            cache_hit_rate_gauge.labels(replica=rid).set(r["cache_hit_rate"])
        for rid, stats in tracker.stats().items():
            cached_prefixes_gauge.labels(replica=rid).set(stats["cached_prefixes"])
        await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Routing proxy
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    messages = body.get("messages", [])
    stream   = body.get("stream", False)

    # --- Select replica ---
    prefix_hash = tracker.hash_prefix(messages)
    result = select_replica(prefix_hash, pool, tracker)

    if result is None:
        return Response(
            content=json.dumps({"error": "No healthy backends available"}),
            status_code=503,
            media_type="application/json",
        )

    replica, is_cache_hit = result
    cache_label = "hit" if is_cache_hit else "miss"

    # Record this routing decision so future requests know this replica
    # now has the prefix cached
    tracker.record_hit(prefix_hash, replica.id)
    pool.acquire(replica.id)

    t_start = time.monotonic()

    try:
        if stream:
            return await _proxy_stream(request, body, replica, cache_label, t_start, prefix_hash)
        else:
            return await _proxy_blocking(request, body, replica, cache_label, t_start, prefix_hash)
    except Exception as exc:
        pool.release(replica.id, ttft_ms=0, cache_hit=False)
        logger.error("Proxy error to %s: %s", replica.id, exc)
        return Response(
            content=json.dumps({"error": str(exc)}),
            status_code=502,
            media_type="application/json",
        )


async def _proxy_blocking(
    request: Request,
    body: dict,
    replica,
    cache_label: str,
    t_start: float,
    prefix_hash: str,
) -> Response:
    url = f"{replica.url}/v1/chat/completions"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    headers["X-KV-Router-Prefix"] = prefix_hash
    headers["X-KV-Router-Cache"]  = cache_label

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=body, headers=headers) as resp:
            ttft_ms = (time.monotonic() - t_start) * 1000
            content = await resp.read()

    pool.release(replica.id, ttft_ms=ttft_ms, cache_hit=(cache_label == "hit"))
    requests_total.labels(replica=replica.id, cache_result=cache_label).inc()
    ttft_histogram.labels(replica=replica.id, cache_result=cache_label).observe(ttft_ms)

    logger.info(
        "→ %s [%s] %.0fms prefix=%s",
        replica.id, cache_label.upper(), ttft_ms, prefix_hash
    )

    return Response(
        content=content,
        status_code=resp.status,
        media_type=resp.content_type,
        headers={"X-Served-By": replica.id, "X-Cache": cache_label},
    )


async def _proxy_stream(
    request: Request,
    body: dict,
    replica,
    cache_label: str,
    t_start: float,
    prefix_hash: str,
) -> StreamingResponse:
    url = f"{replica.url}/v1/chat/completions"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    headers["X-KV-Router-Prefix"] = prefix_hash
    headers["X-KV-Router-Cache"]  = cache_label

    first_chunk = True
    ttft_ms_ref = [0.0]

    async def generate():
        nonlocal first_chunk
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                async for chunk in resp.content.iter_chunked(1024):
                    if first_chunk:
                        ttft_ms_ref[0] = (time.monotonic() - t_start) * 1000
                        first_chunk = False
                    yield chunk
        pool.release(replica.id, ttft_ms=ttft_ms_ref[0], cache_hit=(cache_label == "hit"))
        requests_total.labels(replica=replica.id, cache_result=cache_label).inc()
        ttft_histogram.labels(replica=replica.id, cache_result=cache_label).observe(ttft_ms_ref[0])
        logger.info(
            "→ %s [%s] stream %.0fms prefix=%s",
            replica.id, cache_label.upper(), ttft_ms_ref[0], prefix_hash
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Served-By": replica.id, "X-Cache": cache_label},
    )


# ---------------------------------------------------------------------------
# Admin / observability endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    healthy = [r for r in pool.snapshot() if r["healthy"]]
    return {
        "status": "ok" if healthy else "degraded",
        "backends": pool.snapshot(),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/debug/prefix-stats")
def prefix_stats():
    return tracker.stats()


@app.get("/debug/backends")
def backends():
    return pool.snapshot()
