#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# kv-router — zero-dependency quickstart
#
# What this does:
#   1. Creates a Python virtualenv + installs all deps
#   2. Starts 3 fake LLM backends (simulated vLLM, no GPU needed)
#   3. Starts the KV-cache-aware router
#   4. Runs the full test suite with performance report
#   5. Tears everything down
#
# Requirements: Python 3.8+  (pre-installed on macOS and most Linux distros)
#
# Usage:
#   chmod +x run.sh && ./run.sh
# -----------------------------------------------------------------------------

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

# Ports
BACKEND_PORTS=(8031 8032 8033)
ROUTER_PORT=8030

# Pids to clean up
PIDS=()

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[kv-router]${NC} $*"; }
warn() { echo -e "${YELLOW}[kv-router]${NC} $*"; }
die()  { echo -e "${RED}[kv-router] ERROR:${NC} $*" >&2; exit 1; }

cleanup() {
  echo ""
  warn "Shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  log "Done."
}
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# 1. Python check
# -----------------------------------------------------------------------------

PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    version=$("$cmd" -c "import sys; print(sys.version_info >= (3,8))")
    if [ "$version" = "True" ]; then
      PYTHON="$cmd"
      break
    fi
  fi
done

[ -z "$PYTHON" ] && die "Python 3.8+ is required. Install from https://python.org"
log "Using $($PYTHON --version)"

# -----------------------------------------------------------------------------
# 2. Virtual environment
# -----------------------------------------------------------------------------

if [ ! -d "$VENV" ]; then
  log "Creating virtual environment..."
  "$PYTHON" -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

log "Installing dependencies..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r "$ROOT/router/requirements.txt"
"$PIP" install --quiet pytest

# -----------------------------------------------------------------------------
# 3. Start fake backends
# -----------------------------------------------------------------------------

log "Starting fake LLM backends..."
for i in "${!BACKEND_PORTS[@]}"; do
  port="${BACKEND_PORTS[$i]}"
  PORT=$port BACKEND_ID="backend-$i" \
    "$PYTHON" "$ROOT/simulator/fake_backend.py" \
    > "$ROOT/.backend-$i.log" 2>&1 &
  PIDS+=($!)
done

# -----------------------------------------------------------------------------
# 4. Start router
# -----------------------------------------------------------------------------

log "Starting KV-cache-aware router on :$ROUTER_PORT..."
BACKENDS=$(IFS=,; echo "${BACKEND_PORTS[*]/#/http://localhost:}")
(
  cd "$ROOT/router"
  BACKENDS=$BACKENDS \
    "$PYTHON" -m uvicorn main:app \
    --host 0.0.0.0 --port "$ROUTER_PORT"
) > "$ROOT/.router.log" 2>&1 &
PIDS+=($!)

# -----------------------------------------------------------------------------
# 5. Wait for everything to be ready
# -----------------------------------------------------------------------------

log "Waiting for services to start..."

wait_for() {
  local url=$1 label=$2 timeout=20
  local deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -sf "$url" > /dev/null 2>&1; then
      log "  $label ready"
      return 0
    fi
    sleep 0.5
  done
  die "$label did not start within ${timeout}s — check log files"
}

for i in "${!BACKEND_PORTS[@]}"; do
  wait_for "http://localhost:${BACKEND_PORTS[$i]}/health" "backend-$i"
done
wait_for "http://localhost:$ROUTER_PORT/health" "router"

# -----------------------------------------------------------------------------
# 6. Run tests
# -----------------------------------------------------------------------------

echo ""
log "Running test suite..."
echo ""

ROUTER_PORT=$ROUTER_PORT \
  "$VENV/bin/pytest" tests/test_router.py -v \
  --override-ini="log_cli=false" \
  -p no:warnings \
  --tb=short \
  2>&1

# -----------------------------------------------------------------------------
# 7. Run benchmark
# -----------------------------------------------------------------------------

echo ""
log "Running benchmark (naive vs cache-aware)..."
echo ""

ROUTER_URL="http://localhost:$ROUTER_PORT" \
  "$PYTHON" "$ROOT/simulator/load_gen.py" \
  --mode router --requests 60 --concurrency 5
