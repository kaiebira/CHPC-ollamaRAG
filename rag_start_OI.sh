#!/usr/bin/env bash
# Robust launcher for CHPC RAG + Ollama (fixed ports, Open WebUI-friendly)
# - Starts (or reuses) an Ollama daemon on a FIXED port (default 127.0.0.1:44141)
# - Optional: pre-pull model
# - Starts query_OI.py on :8000 with OpenAI-compatible endpoints
# - Clean shutdown on Ctrl+C

set -Eeuo pipefail
IFS=$'\n\t'

# ---- Config (env overridable) ----
HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-8000}"

# Fixed Ollama port (avoid random ports so Open WebUI can point to RAG reliably)
OLLAMA_ADDR="${OLLAMA_ADDR:-127.0.0.1:44141}"
export OLLAMA_HOST="${OLLAMA_HOST:-$OLLAMA_ADDR}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://$OLLAMA_ADDR}"

# Model + embeddings
MODEL_NAME="${MODEL_NAME:-gpt-oss:20b}"
EMBEDDINGS_MODEL="${EMBEDDINGS_MODEL:-all-mpnet-base-v2}"

# Retrieval
QDRANT_PATH="${QDRANT_PATH:-./langchain_qdrant}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-chpc-rag}"
RETRIEVER_SEARCH="${RETRIEVER_SEARCH:-mmr}"
RETRIEVER_K="${RETRIEVER_K:-5}"
RETRIEVER_FETCH_K="${RETRIEVER_FETCH_K:-10}"
RETRIEVER_SCORE_THRESHOLD="${RETRIEVER_SCORE_THRESHOLD:-0.3}"

# Generation knobs
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.8}"
NUM_CTX="${NUM_CTX:-131072}"

# Health + pulls
WAIT_OLLAMA_SECS="${WAIT_OLLAMA_SECS:-30}"
WAIT_RAG_SECS="${WAIT_RAG_SECS:-30}"
OLLAMA_PULL="${OLLAMA_PULL:-false}"
OLLAMA_MODEL_PULL_NAME="${OLLAMA_MODEL_PULL_NAME:-$MODEL_NAME}"

# Logging
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"
OLLAMA_LOG="$LOG_DIR/ollama.log"
RAG_LOG="$LOG_DIR/rag_server.log"

bold() { tput bold 2>/dev/null || true; }
clr() { tput sgr0 2>/dev/null || true; }
log() { echo -e "$(bold)[$(date +%H:%M:%S)]$(clr) $*"; }
warn(){ echo -e "[WARN] $*" >&2; }
err() { echo -e "[ERR ] $*" >&2; }

need() { command -v "$1" >/dev/null 2>&1 || { err "Missing: $1"; exit 1; }; }
need curl

# Try to load site module if available
if command -v ml >/dev/null 2>&1; then
    ml ollama || true
    # Load apptainer if not already loaded
    if ! command -v apptainer >/dev/null 2>&1; then
        ml apptainer || true
    fi
fi

# ---- Start/reuse Ollama on fixed port ----
is_ollama_up() {
  curl -sf "$OLLAMA_BASE_URL/api/version" >/dev/null
}

if is_ollama_up; then
  log "Ollama already running at $OLLAMA_BASE_URL"
else
  log "Starting Ollama at $OLLAMA_BASE_URL (logging to $OLLAMA_LOG)"
  # Tip: ensure your host Ollama build supports CUDA v12 (Blackwell) on CHPC
  (ollama serve >"$OLLAMA_LOG" 2>&1 &) || { err "Failed to spawn ollama serve"; exit 1; }

  # Wait for health
  t0=$(date +%s)
  until is_ollama_up; do
    if (( $(date +%s) - t0 > WAIT_OLLAMA_SECS )); then
      warn "Ollama not responding within ${WAIT_OLLAMA_SECS}s (continuing; check $OLLAMA_LOG)"
      break
    fi
    sleep 1
  done
fi

# Pre-pull model into THIS daemon
if [[ "$OLLAMA_PULL" == "true" ]]; then
  log "Pulling model: $OLLAMA_MODEL_PULL_NAME"
  if ! ollama pull "$OLLAMA_MODEL_PULL_NAME" >>"$OLLAMA_LOG" 2>&1; then
    warn "Model pull failed (see $OLLAMA_LOG)."
  fi
fi

# ---- Launch RAG app (query_OI.py) ----
export RAG_MODEL="$MODEL_NAME"
export EMBEDDINGS_MODEL="$EMBEDDINGS_MODEL"
export QDRANT_PATH QDRANT_COLLECTION
export RETRIEVER_SEARCH RETRIEVER_K RETRIEVER_FETCH_K RETRIEVER_SCORE_THRESHOLD
export TEMPERATURE TOP_P NUM_CTX
export HTTP_HOST HTTP_PORT
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

log "--------------------------------------------------"
log "RAG server"
log "  Host:              $HTTP_HOST"
log "  Port:              $HTTP_PORT"
log "  Model:             $MODEL_NAME"
log "  Embeddings:        $EMBEDDINGS_MODEL"
log "  Qdrant path:       $QDRANT_PATH"
log "  Collection:        $QDRANT_COLLECTION"
log "  Ollama base URL:   $OLLAMA_BASE_URL"
log "  Context (num_ctx): $NUM_CTX"
log "  Logs:              $RAG_LOG"
log "OpenAI base for WebUI: http://127.0.0.1:${HTTP_PORT}/v1"
log "--------------------------------------------------"

# Launch in background
python3 "$SCRIPT_DIR/query_OI.py" >"$RAG_LOG" 2>&1 &
RAG_PID=$!
log "RAG PID: $RAG_PID"

# Wait for /health
t0=$(date +%s)
until curl -sf "http://127.0.0.1:${HTTP_PORT}/health" >/dev/null; do
  if (( $(date +%s) - t0 > WAIT_RAG_SECS )); then
    warn "RAG not healthy after ${WAIT_RAG_SECS}s (check $RAG_LOG)"
    break
  fi
  sleep 1
done

log "Running. Ctrl+C to stop."
trap 'echo; log "Stopping..."; kill $RAG_PID 2>/dev/null || true; sleep 1; exit 0' INT TERM
wait "$RAG_PID" || true
