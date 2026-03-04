#!/bin/bash
# Start EduBot
# Usage:
#   bash scripts/start.sh          -> Q&A mode (default)
#   bash scripts/start.sh story    -> original story-generation mode
#
# MODE env var also accepted:  MODE=story bash scripts/start.sh

MODE="${1:-${MODE:-qa}}"

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDUBOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INTERFACE_DIR="$EDUBOT_DIR/MI2US_Year2s/Interface"

cd "$INTERFACE_DIR"

# Load .env file if it exists
if [ -f "$EDUBOT_DIR/scripts/.env" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
        export "$line"
    done < "$EDUBOT_DIR/scripts/.env"
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
pkill -f "python.*main.py"      2>/dev/null || true
pkill -f "python.*qa_server.py" 2>/dev/null || true
pkill -f "python.*http.server.*8080" 2>/dev/null || true
sleep 2

echo "[INFO] Starting EduBot in mode: $MODE"

HOST="$(hostname -f 2>/dev/null || hostname)"

if [ "$MODE" = "story" ]; then
    # ---- Legacy story-generation mode ----
    nohup python3 -m http.server 8080 > "$EDUBOT_DIR/logs/http_server.log" 2>&1 &
    HTTP_PID=$!
    echo "$HTTP_PID" > "$EDUBOT_DIR/logs/http_server.pid"
    echo "[SUCCESS] HTTP server started (PID: $HTTP_PID)"

    nohup venv39/bin/python main.py > "$EDUBOT_DIR/logs/edubot.log" 2>&1 &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$EDUBOT_DIR/logs/edubot.pid"
    echo "[SUCCESS] Backend started (PID: $BACKEND_PID)"

    echo ""
    echo "=========================================="
    echo "[SUCCESS] EduBot (story mode) is running!"
    echo "=========================================="
    echo "Open: http://$HOST:8080/storyGeneration.html"
    echo "Logs: $EDUBOT_DIR/logs/"
    echo "Stop: bash scripts/stop.sh"
    echo "=========================================="
else
    # ---- Q&A dialogue mode (default) ----
    # qa_server.py starts the HTTP server internally on port 8080
    nohup venv39/bin/python qa_server.py > "$EDUBOT_DIR/logs/qa_server.log" 2>&1 &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$EDUBOT_DIR/logs/edubot.pid"
    echo "[SUCCESS] Q&A server started (PID: $BACKEND_PID)"

    echo ""
    echo "=========================================="
    echo "[SUCCESS] EduBot (Q&A mode) is running!"
    echo "=========================================="
    echo "Open: http://$HOST:8080/qa.html"
    echo "WebSocket: ws://$HOST:10000"
    echo "Logs: $EDUBOT_DIR/logs/qa_server.log"
    echo "Stop: bash scripts/stop.sh"
    echo "=========================================="
fi
