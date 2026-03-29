# kv-router — KV-Cache-Aware LLM Router

An OpenAI-compatible load balancer that routes LLM inference requests to the replica most likely to have the relevant **KV cache already computed** — eliminating redundant prefill work across a multi-replica serving cluster.

## The Problem

In a standard multi-replica vLLM/SGLang deployment, load balancers use round-robin or least-connections routing. They have no awareness of which replica has already processed a given system prompt or conversation prefix. Result: every replica independently re-computes the same KV blocks from scratch for the same shared prompt.

For a 70B model on an A100:
- **Cold prefill** (no cache): 600–1000ms TTFT for a 512-token system prompt
- **Cache hit** (KV blocks already computed): 80–120ms TTFT

At scale with a shared system prompt across 10,000 requests/hour, that's ~880ms of wasted prefill GPU compute **per request**.

This is the same insight that won Moonshot AI the **FAST 2025 Best Paper** ([Mooncake](https://arxiv.org/abs/2407.00079)) and motivated the [vLLM Router](https://blog.vllm.ai/2025/12/13/vllm-router-release.html) project.

## How It Works

```
Client
  │
  ▼
┌──────────────────────────────────────┐
│         KV-Cache-Aware Router        │
│                                      │
│  1. Hash canonical prefix of msgs    │
│  2. Look up which replica has it     │
│  3. Score: cache_hit_bonus - load    │
│  4. Route to best replica            │
│  5. Record hit for future routing    │
└──────────────────────────────────────┘
         │           │           │
         ▼           ▼           ▼
    backend-0    backend-1    backend-2
    (vLLM)       (vLLM)       (vLLM)
```

**Scoring function:**

```
score(replica) = CACHE_HIT_BONUS × is_cached - LOAD_WEIGHT × in_flight
```

A replica with a warm cache beats a cold idle replica — unless it's so loaded that the queue wait exceeds the cache benefit.

**Prefix hashing:** We hash the system prompt + first N chars of the user message. This captures the stable, expensive-to-compute part of the KV cache while ignoring the per-request tail.

## Project Structure

```
kv-router/
├── router/
│   ├── main.py            # FastAPI proxy, OpenAI-compatible API
│   ├── prefix_tracker.py  # LRU map of prefix_hash → replica
│   ├── backend_pool.py    # Health checks, in-flight tracking
│   ├── routing.py         # Cache-aware scoring algorithm
│   └── requirements.txt
├── simulator/
│   ├── fake_backend.py    # OpenAI-compatible fake vLLM (latency simulation)
│   └── load_gen.py        # Benchmark: naive vs cache-aware routing
├── docker-compose.yml
└── Makefile
```

## Quickstart (local, no GPUs needed)

```bash
# Install deps
pip install -r router/requirements.txt

# Terminal 1: start 3 fake backends
PORT=8001 BACKEND_ID=backend-0 python simulator/fake_backend.py
PORT=8002 BACKEND_ID=backend-1 python simulator/fake_backend.py
PORT=8003 BACKEND_ID=backend-2 python simulator/fake_backend.py

# Terminal 2: start the router
BACKENDS="http://localhost:8001,http://localhost:8002,http://localhost:8003" \
  uvicorn router.main:app --port 8000 --reload

# Terminal 3: run the benchmark
python simulator/load_gen.py --mode both
```

**Expected output:**

```
  Naive (round-robin)
  ────────────────────────────────────────
  Cache hit rate: 0%   (0/60 hits)
  TTFT p50:       812ms
  TTFT p95:       987ms

  Smart (cache-aware)
  ────────────────────────────────────────
  Cache hit rate: 67%  (40/60 hits)
  TTFT p50:       98ms
  TTFT p95:       820ms

  TTFT improvement (p50):  88%
```

## Docker

```bash
docker compose up --build
# then: python simulator/load_gen.py --mode both
```

## Observability

| Endpoint | Description |
|---|---|
| `GET /health` | Replica health + in-flight counts |
| `GET /metrics` | Prometheus metrics |
| `GET /debug/prefix-stats` | Cached prefix counts per replica |
| `GET /debug/backends` | Full backend snapshot |

Prometheus metrics exposed:
- `kv_router_requests_total{replica, cache_result}`
- `kv_router_ttft_milliseconds{replica, cache_result}` (histogram)
- `kv_router_cache_hit_rate{replica}`
- `kv_router_cached_prefixes{replica}`
- `kv_router_in_flight_requests{replica}`

## Connecting to real vLLM

```bash
# Point at real vLLM instances (must have --enable-prefix-caching flag)
BACKENDS="http://vllm-0:8000,http://vllm-1:8000,http://vllm-2:8000" \
  uvicorn router.main:app --port 9000

# Use like any OpenAI-compatible endpoint
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3-70b", "messages": [...]}'
```

## References

- [Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) — FAST 2025 Best Paper
- [vLLM Router](https://blog.vllm.ai/2025/12/13/vllm-router-release.html) — vLLM's own cache-aware router (Rust)
- [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see) — Red Hat / llm-d on distributed KV routing
- [SGLang RFC #7746](https://github.com/sgl-project/sglang/issues/7746) — Remote KV Connector (still open)
