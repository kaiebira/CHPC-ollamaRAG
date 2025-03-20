#!/bin/bash

# Function to show usage
show_usage() {
    echo "Usage:"
    echo "  $0 [--http-host <host>] [--http-port <port>] [--temp <temperature>] [--top-p <top_p>] [--score <score_threshold>] [--chunks <chunks>]"
    echo
    echo "Examples:"
    echo "  # Start server with default parameters"
    echo "  $0"
    echo
    echo "  # Start server with custom parameters"
    echo "  $0 --http-port 8080 --temp 0.3 --top-p 0.5"
    echo
    echo "  # To use streaming with curl:"
    echo "  curl -N -X POST http://localhost:8000/stream -H \"Content-Type: application/json\" -d '{\"prompt\":\"your question here\"}'"
    echo
    echo "  # To check server health:"
    echo "  curl http://localhost:8000/health"
    exit 1
}

# Default values
HTTP_HOST="0.0.0.0"
HTTP_PORT="8000"
TEMPERATURE="0.2"
TOP_P="0.2"
SCORE_THRESHOLD="0.3"
CHUNKS="10"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --http-host) HTTP_HOST="$2"; shift ;;
        --http-port) HTTP_PORT="$2"; shift ;;
        --temp) TEMPERATURE="$2"; shift ;;
        --top-p) TOP_P="$2"; shift ;;
        --score) SCORE_THRESHOLD="$2"; shift ;;
        --chunks) CHUNKS="$2"; shift ;;
        --help) show_usage ;;
        *) echo "Unknown parameter: $1"; show_usage ;;
    esac
    shift
done

# Load ollama module
ml ollama

# Get dynamic port for Ollama
export OLPORT=`ruby -e 'require "socket"; puts Addrinfo.tcp("", 0).bind {|s| s.local_address.ip_port }'`
echo "Using dynamic port for Ollama: $OLPORT"

# Set environment variables to ensure proper Ollama connection
export OLLAMA_HOST="127.0.0.1:$OLPORT"
export OLLAMA_BASE_URL="http://127.0.0.1:$OLPORT"

# Print environment variables for debugging
echo "OLLAMA_HOST: $OLLAMA_HOST"
echo "OLLAMA_BASE_URL: $OLLAMA_BASE_URL"

# Start ollama server in background
echo "Starting Ollama server on port $OLPORT..."
ollama serve >& ollama.log &
OLLAMA_PID=$!
echo "Ollama server started with PID: $OLLAMA_PID"

# Wait for server to start
echo "Waiting for Ollama server to initialize..."
sleep 3

# Test if Ollama is responding
echo "Testing Ollama server..."
OLLAMA_TEST=$(curl -s "$OLLAMA_BASE_URL/api/tags" || echo "Connection failed")
if [[ "$OLLAMA_TEST" == *"Connection failed"* ]]; then
    echo "WARNING: Ollama server test failed. Check ollama.log for details."
    echo "Attempting to continue anyway..."
else
    echo "Ollama server is responding."
fi

# Save Ollama PID to a file for cleanup
echo $OLLAMA_PID > ollama.pid

# Set up trap to clean up Ollama server on exit
cleanup() {
    echo "Cleaning up..."
    if [ -f ollama.pid ]; then
        OLLAMA_PID=$(cat ollama.pid)
        echo "Stopping Ollama server (PID: $OLLAMA_PID)..."
        kill $OLLAMA_PID 2>/dev/null || true
        rm ollama.pid
    fi
    exit 0
}
trap cleanup INT TERM EXIT

# Start the CHPC RAG server
echo "Starting HTTP server on $HTTP_HOST:$HTTP_PORT..."
echo "Regular endpoint: http://$HTTP_HOST:$HTTP_PORT/"
echo "Streaming endpoint: http://$HTTP_HOST:$HTTP_PORT/stream"
echo "Health check endpoint: http://$HTTP_HOST:$HTTP_PORT/health"
echo 
echo "Example curl command for streaming:"
echo "curl -N -X POST http://$HTTP_HOST:$HTTP_PORT/stream \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"prompt\":\"How do I submit a job to CHPC?\"}'"
echo
echo "Example curl command for health check:"
echo "curl http://$HTTP_HOST:$HTTP_PORT/health"

python query.py -t "$TEMPERATURE" -p "$TOP_P" -s "$SCORE_THRESHOLD" -c "$CHUNKS" \
  --host "$HTTP_HOST" --port "$HTTP_PORT" --ollama-url "$OLLAMA_BASE_URL"
