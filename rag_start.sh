#!/bin/bash

# ==============================================================================
# CHPC RAG Assistant Server Startup Script (`run.sh`)
# ==============================================================================
#
# Purpose:
#   Starts the Ollama LLM server and the Python-based RAG query server (`query.py`).
#   Handles dynamic port allocation for Ollama, environment variable setup,
#   and cleanup of the Ollama process on exit.
#
# Dependencies:
#   - `ml` command (or similar module loading system)
#   - `ollama` installed and available via `ml ollama`
#   - `ruby` (for dynamic port finding)
#   - `python` (to run query.py)
#   - `query.py` script in the same directory or accessible via Python path
#   - Required Python packages for query.py installed
#
# ==============================================================================

# --- Default Configuration ---
# These defaults align with the Settings class in query.py but can be overridden
# by command-line arguments below.
HTTP_HOST="0.0.0.0"
HTTP_PORT="8000"
TEMPERATURE="0.2"
TOP_P="0.8"
SCORE_THRESHOLD="0.3"
CHUNKS="5" # Corresponds to retriever_k in query.py settings
LOG_LEVEL="INFO"
MODEL_NAME="gemma3:4b"
QDRANT_PATH="./langchain_qdrant"
COLLECTION_NAME="chpc-rag"
# OLLAMA_BASE_URL will be set dynamically

# --- Usage Function ---
# Displays help message and exits.
show_usage() {
    echo "Usage:"
    echo "  $0 [options]"
    echo
    echo "Options:"
    echo "  --http-host <host>         Host for the RAG server (Default: ${HTTP_HOST})"
    echo "  --http-port <port>         Port for the RAG server (Default: ${HTTP_PORT})"
    echo "  --temperature <temp>       LLM temperature (Default: ${TEMPERATURE})"
    echo "  --top-p <top_p>            LLM top-p (Default: ${TOP_P})"
    echo "  --score-threshold <score>  Retriever score threshold (Default: ${SCORE_THRESHOLD})"
    echo "  --chunks <num>             Number of chunks to retrieve (Default: ${CHUNKS})"
    echo "  --log-level <level>        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (Default: ${LOG_LEVEL})"
    echo "  --model <name>             Ollama model name (Default: ${MODEL_NAME})"
    echo "  --qdrant-path <path>       Path to Qdrant data directory (Default: ${QDRANT_PATH})"
    echo "  --collection <name>        Qdrant collection name (Default: ${COLLECTION_NAME})"
    echo "  --help                     Show this help message"
    echo
    echo "Examples:"
    echo "  # Start server with default parameters"
    echo "  $0"
    echo
    echo "  # Start server with custom parameters"
    echo "  $0 --http-port 8080 --temperature 0.3 --log-level DEBUG"
    echo
    echo "  # To use streaming with curl (replace host/port if changed):"
    echo "  curl -N -X POST http://localhost:8000/stream -H \"Content-Type: application/json\" -d '{\"prompt\":\"your question here\"}'"
    echo
    echo "  # To check server health:"
    echo "  curl http://localhost:8000/health"
    exit 1
}

# --- Argument Parsing ---
# Parses command-line arguments to override default settings.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        # Server arguments
        --http-host) HTTP_HOST="$2"; shift ;;
        --http-port) HTTP_PORT="$2"; shift ;;
        # LLM/Retrieval arguments (matching query.py arg names where possible)
        --temperature) TEMPERATURE="$2"; shift ;;
        --top-p) TOP_P="$2"; shift ;;
        --score-threshold) SCORE_THRESHOLD="$2"; shift ;;
        --chunks) CHUNKS="$2"; shift ;;
        # New arguments matching query.py settings/args
        --log-level) LOG_LEVEL="$2"; shift ;;
        --model) MODEL_NAME="$2"; shift ;;
        --qdrant-path) QDRANT_PATH="$2"; shift ;;
        --collection) COLLECTION_NAME="$2"; shift ;;
        # Help argument
        --help) show_usage ;;
        # Handle unknown arguments
        *) echo "Unknown parameter: $1"; show_usage ;;
    esac
    shift # Move to the next argument pair
done

# --- Ollama Setup ---
# Load the necessary environment module for Ollama. Adjust if your system differs.
echo "Loading Ollama module..."
ml ollama

# Find a dynamic, available port for the Ollama server using Ruby.
# This avoids port conflicts if multiple instances are run.
echo "Finding dynamic port for Ollama..."
export OLPORT=$(ruby -e 'require "socket"; puts Addrinfo.tcp("", 0).bind {|s| s.local_address.ip_port }')
if [ -z "$OLPORT" ]; then
    echo "ERROR: Failed to get a dynamic port for Ollama. Exiting."
    exit 1
fi
echo "Using dynamic port for Ollama: $OLPORT"

# Set environment variables required by Ollama and query.py to connect.
# OLLAMA_HOST is used by the 'ollama serve' command itself.
# OLLAMA_BASE_URL is used by the LangChain OllamaLLM client in query.py.
export OLLAMA_HOST="127.0.0.1:$OLPORT"
export OLLAMA_BASE_URL="http://127.0.0.1:$OLPORT"

# Print environment variables for debugging purposes.
echo "OLLAMA_HOST set to: $OLLAMA_HOST"
echo "OLLAMA_BASE_URL set to: $OLLAMA_BASE_URL"

# Start the Ollama server in the background.
# Redirect stdout and stderr to a log file.
echo "Starting Ollama server on port $OLPORT..."
ollama serve >& ollama.log &
OLLAMA_PID=$! # Capture the Process ID (PID) of the background Ollama server.
echo "Ollama server started with PID: $OLLAMA_PID"

# Brief pause to allow the Ollama server time to initialize.
echo "Waiting for Ollama server to initialize (3 seconds)..."
sleep 3

# Test if the Ollama server is responding.
# This is a basic check; the server might still be loading models.
echo "Testing Ollama server connection at $OLLAMA_BASE_URL..."
# Use curl with silent (-s) and fail (-f) flags for a cleaner check.
if curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null; then
    echo "Ollama server is responding."
else
    echo "WARNING: Ollama server test failed. Check ollama.log for details."
    echo "Attempting to continue anyway..."
    # Consider adding more robust checks or longer waits if needed.
fi

# Save the Ollama PID to a file for the cleanup function.
echo $OLLAMA_PID > ollama.pid

# --- Cleanup Function ---
# Defines actions to take when the script receives an interrupt (INT),
# termination (TERM), or exit (EXIT) signal.
cleanup() {
    echo # Newline for cleaner output
    echo "Cleaning up..."
    if [ -f ollama.pid ]; then
        OLLAMA_PID_TO_KILL=$(cat ollama.pid)
        if ps -p $OLLAMA_PID_TO_KILL > /dev/null; then
           echo "Stopping Ollama server (PID: $OLLAMA_PID_TO_KILL)..."
           # Send SIGTERM first, then SIGKILL if it doesn't stop
           kill $OLLAMA_PID_TO_KILL 2>/dev/null
           sleep 1 # Give it a moment to shut down
           kill -9 $OLLAMA_PID_TO_KILL 2>/dev/null || true # Force kill if still running
        else
           echo "Ollama server (PID: $OLLAMA_PID_TO_KILL) not found or already stopped."
        fi
        rm ollama.pid # Remove the PID file
    else
        echo "Ollama PID file not found."
    fi
    echo "Cleanup complete."
    exit 0 # Exit cleanly
}

# Register the cleanup function to run on specific signals.
trap cleanup INT TERM EXIT

# --- Start RAG Server (query.py) ---
# Executes the main Python application script, passing the configured parameters.
echo "--------------------------------------------------"
echo "Starting RAG HTTP server (query.py)..."
echo "Host:            ${HTTP_HOST}"
echo "Port:            ${HTTP_PORT}"
echo "Log Level:       ${LOG_LEVEL}"
echo "Model:           ${MODEL_NAME}"
echo "Temperature:     ${TEMPERATURE}"
echo "Top-P:           ${TOP_P}"
echo "Score Threshold: ${SCORE_THRESHOLD}"
echo "Chunks (k):      ${CHUNKS}"
echo "Qdrant Path:     ${QDRANT_PATH}"
echo "Collection:      ${COLLECTION_NAME}"
echo "Ollama URL:      ${OLLAMA_BASE_URL}"
echo "--------------------------------------------------"
echo "Endpoints:"
echo "  Regular:       http://${HTTP_HOST}:${HTTP_PORT}/"
echo "  Streaming:     http://${HTTP_HOST}:${HTTP_PORT}/stream"
echo "  Health Check:  http://${HTTP_HOST}:${HTTP_PORT}/health"
echo "--------------------------------------------------"
echo "To stop the server, press Ctrl+C."

# Run the Python script, passing arguments.
# Note: Argument names here MUST match those defined in query.py's argparse setup.
python query.py \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --score-threshold "$SCORE_THRESHOLD" \
    --chunks "$CHUNKS" \
    --host "$HTTP_HOST" \
    --port "$HTTP_PORT" \
    --ollama-url "$OLLAMA_BASE_URL" \
    --log-level "$LOG_LEVEL" \
    --model "$MODEL_NAME" \
    --qdrant-path "$QDRANT_PATH" \
    --collection "$COLLECTION_NAME"

# The script will wait here until query.py finishes or the script is interrupted.
# The trap ensures cleanup happens upon exit.
echo "RAG server (query.py) has stopped."

