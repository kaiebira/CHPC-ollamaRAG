#!/bin/bash

ml ollama/0.3.14
export OLPORT=`ruby -e 'require "socket"; puts Addrinfo.tcp("", 0).bind {|s| s.local_address.ip_port }'`
export OLLAMA_HOST=127.0.0.1:$OLPORT
export OLLAMA_BASE_URL="http://localhost:$OLPORT"
ollama serve >& ollama.log &
sleep 1
python query.py -S -M
