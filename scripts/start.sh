#!/bin/bash
# Start EduBot - Simple startup script
# Usage: bash scripts/start.sh

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDUBOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INTERFACE_DIR="$EDUBOT_DIR/MI2US_Year2s/Interface"

cd "$INTERFACE_DIR"

# Load .env file if it exists
if [ -f "$EDUBOT_DIR/scripts/.env" ]; then
    set -a
    source "$EDUBOT_DIR/scripts/.env"
    set +a
    echo "[INFO] Loaded environment variables from .env"
fi

# Set cache directories (relative to EduBot)
export PIP_CACHE_DIR="$EDUBOT_DIR/.cache/pip"
export TMPDIR="$EDUBOT_DIR/tmp"
export HF_HOME="$EDUBOT_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$EDUBOT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$EDUBOT_DIR/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0

# Create cache directories
mkdir -p "$EDUBOT_DIR/.cache/huggingface" "$EDUBOT_DIR/.cache/pip" "$EDUBOT_DIR/tmp" "$EDUBOT_DIR/logs"

# Set OpenAI API key (from .env or fallback)
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f "$EDUBOT_DIR/scripts/.env" ]; then
        export OPENAI_API_KEY=$(grep OPENAI_API_KEY "$EDUBOT_DIR/scripts/.env" | cut -d '=' -f2 | tr -d '"' | tr -d "'")
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv39" ]; then
    echo "[ERROR] Virtual environment not found. Please run setup first."
    echo "  Run: bash scripts/setup.sh"
    exit 1
fi

# Kill any existing instances
echo "[INFO] Stopping any existing EduBot instances..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "python.*http.server.*8080" 2>/dev/null || true
sleep 2

echo "[INFO] Starting EduBot..."

# Start HTTP server on port 8080
nohup python3 -m http.server 8080 > "$EDUBOT_DIR/logs/http_server.log" 2>&1 &
HTTP_PID=$!
echo "$HTTP_PID" > "$EDUBOT_DIR/logs/http_server.pid"
echo "[SUCCESS] HTTP server started (PID: $HTTP_PID)"
echo "         Access: http://$(hostname -f 2>/dev/null || hostname):8080/storyGeneration.html"

# Start Python backend on port 10000
nohup venv39/bin/python main.py > "$EDUBOT_DIR/logs/edubot.log" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$EDUBOT_DIR/logs/edubot.pid"
echo "[SUCCESS] Backend started (PID: $BACKEND_PID)"
echo "         WebSocket: ws://$(hostname -f 2>/dev/null || hostname):10000"

echo ""
echo "=========================================="
echo "[SUCCESS] EduBot is running!"
echo "=========================================="
echo "HTTP Server: http://$(hostname -f 2>/dev/null || hostname):8080/storyGeneration.html"
echo "Logs: $EDUBOT_DIR/logs/"
echo "Stop: bash scripts/stop.sh"
echo "=========================================="
