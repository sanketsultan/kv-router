# kv-router — KV-Cache-Aware LLM Router

![CI](https://github.com/sanketsultan/kv-router/actions/workflows/ci.yml/badge.svg)

An OpenAI-compatible load balancer that routes LLM inference requests to the replica most likely to have the relevant **KV cache already computed** — eliminating redundant prefill work across a multi-replica serving cluster.

## The Problem

In a standard multi-replica vLLM/SGLang deployment, load balancers use round-robin or least-connections routing. They have no awareness of which replica has already processed a given system prompt. Result: every replica independently re-computes the same KV blocks from scratch for every request, even when thousands of requests share the same system prompt.

For a 70B model on an A100:
- **Cold prefill** (cache miss): 600–1000ms TTFT for a 512-token system prompt
- **Cache hit** (KV blocks already computed): 80–120ms TTFT

At scale with a shared system prompt across 10,000 requests/hour, that's ~880ms of wasted prefill GPU compute **per request**.

This is the same insight behind Moonshot AI's **FAST 2025 Best Paper** ([Mooncake](https://arxiv.org/abs/2407.00079)) and the [vLLM Router](https://blog.vllm.ai/2025/12/13/vllm-router-release.html) project.

## Benchmark Results

Tested against simulated A100 latency (80–120ms hit, 600–1000ms miss):

```
  Naive (round-robin)          Smart (cache-aware)
  ──────────────────────────   ──────────────────────────
  Cache hit rate:  0%          Cache hit rate:  67–75%
  TTFT p50:        812ms       TTFT p50:        98ms
  TTFT p95:        987ms       TTFT p95:        820ms

  TTFT improvement (p50): 86–88%
```

Tested against real **llama3.2:1b via Ollama**:

```
  Scenario              Requests   Hit Rate   p50 TTFT
  ──────────────────────────────────────────────────────
  Finance (Basel III)       10       100%      10,590ms
  Medical                    6        83%       7,936ms
  Coding (Python)            6        83%      21,217ms
  ──────────────────────────────────────────────────────
  Overall                   22        91%

  HIT p50:  8,658ms  |  MISS p50: 19,292ms  |  Improvement: 55%
```

## How It Works

```
Client
  │
  ▼
┌──────────────────────────────────────┐
│         KV-Cache-Aware Router        │
│                                      │
│  1. Hash system prompt (stable       │
│     prefix — what vLLM caches)       │
│  2. Look up which replica has it     │
│  3. Score: cache_hit_bonus - load    │
│  4. Route to best replica            │
│  5. Record routing for future reqs   │
└──────────────────────────────────────┘
         │           │           │
         ▼           ▼           ▼
    backend-0    backend-1    backend-2
    (vLLM)       (vLLM)       (vLLM)
```

**Cache key:** The system prompt (and any prior conversation turns). The final user message is excluded — it's unique per request. This matches exactly what vLLM's internal prefix cache stores as KV blocks.

**Scoring function:**
```
score(replica) = CACHE_HIT_BONUS × is_cached - LOAD_WEIGHT × in_flight
```

A replica with a warm cache beats a cold idle one — unless it's so loaded the queue wait exceeds the cache benefit. The crossover is intentional: prevents thundering-herd on a single warmed replica.

## Project Structure

```
kv-router/
├── router/
│   ├── main.py            # FastAPI proxy, OpenAI-compatible API
│   ├── prefix_tracker.py  # LRU map: prefix_hash → {replica: last_seen}
│   ├── backend_pool.py    # Health checks, in-flight tracking, TTFT EMA
│   ├── routing.py         # Cache-aware scoring algorithm
│   └── requirements.txt
├── simulator/
│   ├── fake_backend.py    # Fake vLLM: 80–120ms on hit, 600–1000ms on miss
│   └── load_gen.py        # Benchmark: naive round-robin vs cache-aware
├── tests/
│   ├── conftest.py        # Session fixtures (fake stack + Ollama stack)
│   ├── test_router.py     # 9 automated tests (fake backends, ~20s)
│   └── test_ollama.py     # 5 real-LLM tests (requires ollama serve)
├── run.sh                 # Zero-dependency single-command quickstart
├── docker-compose.yml
└── Makefile
```

## Quickstart — single command, no setup

Requires only Python 3.8+:

```bash
git clone https://github.com/sanketsultan/kv-router.git
cd kv-router
./run.sh
```

This creates a virtualenv, installs deps, starts 3 fake backends + the router, runs 9 automated tests with a performance report, runs a 60-request benchmark, then tears everything down.

## Manual setup

```bash
pip3 install -r router/requirements.txt

# Terminal 1–3: fake backends
PORT=8001 BACKEND_ID=backend-0 python3 simulator/fake_backend.py
PORT=8002 BACKEND_ID=backend-1 python3 simulator/fake_backend.py
PORT=8003 BACKEND_ID=backend-2 python3 simulator/fake_backend.py

# Terminal 4: router
cd router
BACKENDS="http://localhost:8001,http://localhost:8002,http://localhost:8003" \
  uvicorn main:app --port 8000 --reload

# Terminal 5: benchmark
python3 simulator/load_gen.py --mode both
```

## Tests

```bash
# Fast tests — fake backends, no GPU (~20s)
pytest tests/test_router.py -v

# Real LLM tests — requires: ollama serve && ollama pull llama3.2:1b
pytest tests/test_ollama.py -v -s

# All tests + performance report
pytest tests/ -v
```

## Testing with Ollama (real model, no GPU)

```bash
brew install ollama
ollama serve &
ollama pull llama3.2:1b

cd router
BACKENDS="http://localhost:11434,http://localhost:11434,http://localhost:11434" \
HEALTH_PATH="/" \
  uvicorn main:app --port 8000 --reload

pytest tests/test_ollama.py -v -s
```

## Connecting to real vLLM

```bash
# vLLM instances must be started with --enable-prefix-caching
BACKENDS="http://vllm-0:8000,http://vllm-1:8000,http://vllm-2:8000" \
  uvicorn router.main:app --port 9000

curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3-70b", "messages": [...]}'
```

## Observability

| Endpoint | Description |
|---|---|
| `GET /health` | Replica health + in-flight counts |
| `GET /metrics` | Prometheus metrics |
| `GET /debug/prefix-stats` | Cached prefix counts per replica |
| `GET /debug/backends` | Full backend snapshot |

Prometheus metrics:
- `kv_router_requests_total{replica, cache_result}` — hit/miss counter
- `kv_router_ttft_milliseconds{replica, cache_result}` — TTFT histogram
- `kv_router_cache_hit_rate{replica}` — rolling hit rate per replica
- `kv_router_cached_prefixes{replica}` — live prefix count per replica
- `kv_router_in_flight_requests{replica}` — current load per replica

## References

- [Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) — FAST 2025 Best Paper
- [vLLM Router](https://blog.vllm.ai/2025/12/13/vllm-router-release.html) — vLLM's cache-aware router (Rust, Dec 2025)
- [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see) — Red Hat / llm-d on distributed KV routing
- [SGLang RFC #7746](https://github.com/sgl-project/sglang/issues/7746) — Remote KV Connector (still open)
- [DistServe OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) — Disaggregated prefill/decode serving
