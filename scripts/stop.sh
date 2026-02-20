#!/bin/bash
# Stop EduBot
# Usage: bash scripts/stop.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDUBOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[INFO] Stopping EduBot..."

# Kill by PID if saved
if [ -f "$EDUBOT_DIR/logs/http_server.pid" ]; then
    kill $(cat "$EDUBOT_DIR/logs/http_server.pid") 2>/dev/null || true
    rm "$EDUBOT_DIR/logs/http_server.pid"
fi

if [ -f "$EDUBOT_DIR/logs/edubot.pid" ]; then
    kill $(cat "$EDUBOT_DIR/logs/edubot.pid") 2>/dev/null || true
    rm "$EDUBOT_DIR/logs/edubot.pid"
fi

# Kill by process name (backup)
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "python.*http.server.*8080" 2>/dev/null || true

echo "[SUCCESS] EduBot stopped"
