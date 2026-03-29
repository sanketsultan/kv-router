"""
Fake vLLM backend — OpenAI-compatible inference server with simulated
KV-cache-aware latency.

The key behaviour:
  - If the incoming request carries X-KV-Router-Cache: hit, TTFT is SHORT
    (simulates a cache hit — no KV blocks need to be recomputed)
  - Otherwise TTFT is LONG (cold prefill: GPU must process every token)

This makes the cache hit improvement MEASURABLE locally without real GPUs.

Latency model (approximate real-world numbers for a 70B model on A100):
  - Cache HIT  TTFT:  80–120 ms  (just the decode, no prefill)
  - Cache MISS TTFT:  600–1000 ms (full prefill for a ~512-token prompt)

Run multiple instances on different ports:
  PORT=8001 python fake_backend.py
  PORT=8002 python fake_backend.py
  PORT=8003 python fake_backend.py
"""

import asyncio
import json
import os
import random
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

PORT = int(os.getenv("PORT", "8001"))
BACKEND_ID = os.getenv("BACKEND_ID", f"backend-{PORT}")

# Latency parameters (ms)
HIT_TTFT_MIN  = int(os.getenv("HIT_TTFT_MIN",  "80"))
HIT_TTFT_MAX  = int(os.getenv("HIT_TTFT_MAX",  "120"))
MISS_TTFT_MIN = int(os.getenv("MISS_TTFT_MIN", "600"))
MISS_TTFT_MAX = int(os.getenv("MISS_TTFT_MAX", "1000"))

# Simulated token generation rate: tokens/sec after first token
TOKENS_PER_SEC = float(os.getenv("TOKENS_PER_SEC", "40.0"))


def _make_completion(prompt_tokens: int, content: str, backend_id: str) -> dict:
    completion_tokens = len(content.split())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "fake-llm-70b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "x_backend": backend_id,
    }


def _stream_chunks(content: str, backend_id: str):
    """Yield SSE chunks for a streaming response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    words = content.split()
    for word in words:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "fake-llm-70b",
            "choices": [{
                "index": 0,
                "delta": {"content": word + " "},
                "finish_reason": None,
            }],
            "x_backend": backend_id,
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()

    # Final chunk
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "fake-llm-70b",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n".encode()
    yield b"data: [DONE]\n\n"


class FakeBackendHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "backend": BACKEND_ID})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._respond(404, {"error": "not found"})
            return

        length  = int(self.headers.get("Content-Length", 0))
        body    = json.loads(self.rfile.read(length))
        cache_hint = self.headers.get("X-KV-Router-Cache", "miss")
        stream  = body.get("stream", False)
        messages = body.get("messages", [])

        # Estimate prompt tokens (rough: 1 token ≈ 4 chars)
        total_chars   = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = max(1, total_chars // 4)

        # Simulate TTFT based on cache status
        if cache_hint == "hit":
            ttft_ms = random.uniform(HIT_TTFT_MIN, HIT_TTFT_MAX)
        else:
            ttft_ms = random.uniform(MISS_TTFT_MIN, MISS_TTFT_MAX)

        time.sleep(ttft_ms / 1000.0)  # simulate prefill latency

        # Generate a plausible response
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "Hello"
        )
        response_text = (
            f"This is a simulated response from {BACKEND_ID}. "
            f"Your message was: '{last_user_msg[:80]}'. "
            f"Cache status: {cache_hint}. "
            f"Prefill took {ttft_ms:.0f}ms."
        )

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Backend", BACKEND_ID)
            self.send_header("X-TTFT-Ms", f"{ttft_ms:.0f}")
            self.end_headers()

            delay_per_word = 1.0 / TOKENS_PER_SEC
            for chunk in _stream_chunks(response_text, BACKEND_ID):
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(delay_per_word)
        else:
            completion = _make_completion(prompt_tokens, response_text, BACKEND_ID)
            self._respond(200, completion, extra_headers={
                "X-Backend": BACKEND_ID,
                "X-TTFT-Ms": f"{ttft_ms:.0f}",
            })

    def _respond(self, code: int, body: dict, extra_headers: dict = None):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        cache = self.headers.get("X-KV-Router-Cache", "?") if hasattr(self, "headers") else "?"
        print(f"[{BACKEND_ID}] {fmt % args}  cache={cache}")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), FakeBackendHandler)
    print(f"[{BACKEND_ID}] Fake vLLM backend on :{PORT}")
    print(f"  Cache HIT  TTFT: {HIT_TTFT_MIN}–{HIT_TTFT_MAX}ms")
    print(f"  Cache MISS TTFT: {MISS_TTFT_MIN}–{MISS_TTFT_MAX}ms")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[{BACKEND_ID}] Stopped.")
