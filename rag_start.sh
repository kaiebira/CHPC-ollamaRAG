#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <arg1> <arg2> <arg3> <arg4>"
    exit 1
fi

# Load ollama module
ml ollama/0.3.14

# Get dynamic port
export OLPORT=`ruby -e 'require "socket"; puts Addrinfo.tcp("", 0).bind {|s| s.local_address.ip_port }'`
export OLLAMA_HOST=127.0.0.1:$OLPORT
export OLLAMA_BASE_URL="http://localhost:$OLPORT"

# Start ollama server in background
ollama serve >& ollama.log &

# Wait for server to start
sleep 1

# Run Python script with command line arguments
python query.py -t "$1" -p "$2" -s "$3" -c "$4"
