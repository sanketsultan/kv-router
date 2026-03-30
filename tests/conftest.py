"""
Shared pytest fixtures and terminal summary hooks.

Two fixtures:
  running_stack  — fake backends + router (no GPU, fast)
  ollama_stack   — real Ollama LLM + router (requires: ollama serve)
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Fake-backend stack
# ---------------------------------------------------------------------------

FAKE_BACKENDS = [
    {"id": "backend-0", "port": 8011},
    {"id": "backend-1", "port": 8012},
    {"id": "backend-2", "port": 8013},
]
FAKE_ROUTER_PORT  = 8010
FAKE_ROUTER_URL   = f"http://localhost:{FAKE_ROUTER_PORT}"
FAKE_BACKEND_URLS = [f"http://localhost:{b['port']}" for b in FAKE_BACKENDS]

_fake_metrics: dict = {}


def _wait_for(url: str, timeout: int = 20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"{url} did not become ready within {timeout}s")


@pytest.fixture(scope="session")
def running_stack():
    procs = []
    for b in FAKE_BACKENDS:
        env = os.environ.copy()
        env["PORT"]       = str(b["port"])
        env["BACKEND_ID"] = b["id"]
        p = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "simulator", "fake_backend.py")],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)

    backends_env = ",".join(f"http://localhost:{b['port']}" for b in FAKE_BACKENDS)
    env = os.environ.copy()
    env["BACKENDS"] = backends_env
    router_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", str(FAKE_ROUTER_PORT)],
        cwd=os.path.join(ROOT, "router"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    procs.append(router_proc)

    for b in FAKE_BACKENDS:
        _wait_for(f"http://localhost:{b['port']}/health")
    _wait_for(f"{FAKE_ROUTER_URL}/health")

    yield {
        "router_url":   FAKE_ROUTER_URL,
        "backend_urls": FAKE_BACKEND_URLS,
        "metrics":      _fake_metrics,
    }

    for p in procs:
        p.terminate()
    for p in procs:
        p.wait(timeout=5)


# ---------------------------------------------------------------------------
# Ollama stack
# ---------------------------------------------------------------------------

OLLAMA_URL         = os.getenv("OLLAMA_URL",         "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",       "llama3.2:1b")
OLLAMA_ROUTER_PORT = int(os.getenv("OLLAMA_ROUTER_PORT", "8020"))
OLLAMA_ROUTER_URL  = f"http://localhost:{OLLAMA_ROUTER_PORT}"

_ollama_metrics: dict = {}


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=2)
        return True
    except Exception:
        return False


def _model_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
        models = [m["name"].split(":")[0] for m in data.get("models", [])]
        return OLLAMA_MODEL.split(":")[0] in models
    except Exception:
        return False


@pytest.fixture(scope="session")
def ollama_stack():
    if not _ollama_running():
        pytest.skip("Ollama not running — start with: ollama serve")
    if not _model_available():
        pytest.skip(f"Model {OLLAMA_MODEL} not pulled — run: ollama pull {OLLAMA_MODEL}")

    backends_env = ",".join([OLLAMA_URL] * 3)
    env = os.environ.copy()
    env["BACKENDS"]    = backends_env
    env["HEALTH_PATH"] = "/"

    router_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", str(OLLAMA_ROUTER_PORT)],
        cwd=os.path.join(ROOT, "router"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    _wait_for(f"{OLLAMA_ROUTER_URL}/health")

    yield {
        "router_url": OLLAMA_ROUTER_URL,
        "ollama_url": OLLAMA_URL,
        "model":      OLLAMA_MODEL,
        "metrics":    _ollama_metrics,
    }

    router_proc.terminate()
    router_proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    _print_fake_report(terminalreporter, _fake_metrics)
    _print_ollama_report(terminalreporter, _ollama_metrics)


def _print_fake_report(tr, m: dict):
    if not m:
        return
    tr.write_sep("=", "KV Router (Fake Backends) — Performance Report")

    if "backend_health" in m:
        tr.write_line("\n  Backends")
        tr.write_line("  " + "─" * 44)
        for b in m["backend_health"]:
            status = "healthy" if b["healthy"] else "DOWN"
            tr.write_line(f"  {b['id']:12s}  {b['url']:30s}  {status}")

    if "hit_rate" in m:
        tr.write_line("\n  Cache Routing")
        tr.write_line("  " + "─" * 44)
        tr.write_line(f"  Requests sent:     {m.get('total_requests', '?')}")
        tr.write_line(f"  Cache hit rate:    {m['hit_rate']:.0%}  ({m.get('hits', '?')}/{m.get('total_requests', '?')})")

    if "hit_p50_ms" in m and "miss_p50_ms" in m:
        improvement = (m["miss_p50_ms"] - m["hit_p50_ms"]) / m["miss_p50_ms"] * 100
        tr.write_line("\n  TTFT (Time to First Token)")
        tr.write_line("  " + "─" * 44)
        tr.write_line(f"  Cache HIT  p50:    {m['hit_p50_ms']:.0f}ms")
        tr.write_line(f"  Cache HIT  p95:    {m.get('hit_p95_ms', 0):.0f}ms")
        tr.write_line(f"  Cache MISS p50:    {m['miss_p50_ms']:.0f}ms")
        tr.write_line(f"  Cache MISS p95:    {m.get('miss_p95_ms', 0):.0f}ms")
        tr.write_line(f"  Improvement (p50): {improvement:.1f}%  ← headline number")

    if "concurrency_total" in m:
        tr.write_line("\n  Concurrency")
        tr.write_line("  " + "─" * 44)
        tr.write_line(f"  Requests:          {m['concurrency_success']}/{m['concurrency_total']} succeeded")
        tr.write_line(f"  Workers:           {m.get('concurrency_workers', '?')}")

    tr.write_line("")


def _print_ollama_report(tr, m: dict):
    if not m:
        return
    tr.write_sep("=", "KV Router + Ollama (Real LLM) — Performance Report")
    tr.write_line(f"\n  Model:   {m.get('model', OLLAMA_MODEL)}")
    tr.write_line(f"  Backend: {OLLAMA_URL}")

    if "scenarios" in m:
        tr.write_line("\n  Scenario Results")
        tr.write_line("  " + "─" * 52)
        tr.write_line(f"  {'Scenario':<25} {'Requests':>8} {'Hit Rate':>10} {'p50 ms':>8} {'p95 ms':>8}")
        tr.write_line("  " + "─" * 52)
        for s in m["scenarios"]:
            tr.write_line(
                f"  {s['name']:<25} {s['total']:>8} "
                f"{s['hit_rate']:>9.0%} {s['p50_ms']:>8.0f} {s['p95_ms']:>8.0f}"
            )
        tr.write_line("  " + "─" * 52)
        total_reqs = sum(s["total"] for s in m["scenarios"])
        total_hits = sum(s["hits"]  for s in m["scenarios"])
        overall    = total_hits / total_reqs if total_reqs else 0
        tr.write_line(f"  {'TOTAL':<25} {total_reqs:>8} {overall:>9.0%}")

    if m.get("hit_ms") and m.get("miss_ms"):
        hit_p50  = sorted(m["hit_ms"])[len(m["hit_ms"]) // 2]
        miss_p50 = sorted(m["miss_ms"])[len(m["miss_ms"]) // 2]
        if miss_p50 > 0:
            improvement = (miss_p50 - hit_p50) / miss_p50 * 100
            tr.write_line("\n  TTFT: HIT vs MISS (real LLM)")
            tr.write_line("  " + "─" * 44)
            tr.write_line(f"  HIT  p50:    {hit_p50:.0f}ms")
            tr.write_line(f"  MISS p50:    {miss_p50:.0f}ms")
            tr.write_line(f"  Improvement: {improvement:.1f}%")

    tr.write_line("")
