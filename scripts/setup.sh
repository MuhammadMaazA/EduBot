#!/bin/bash
# Setup EduBot - Install dependencies and create virtual environment
# Usage: bash scripts/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDUBOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INTERFACE_DIR="$EDUBOT_DIR/MI2US_Year2s/Interface"

cd "$INTERFACE_DIR"

echo "[INFO] Setting up EduBot..."
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "[INFO] Python version: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv39" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv39
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv39/bin/activate

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing requirements..."
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt not found, installing basic packages..."
    pip install torch transformers peft accelerate openai websockets
fi

# Create necessary directories
mkdir -p "$EDUBOT_DIR/logs" "$EDUBOT_DIR/.cache/huggingface" "$EDUBOT_DIR/.cache/pip" "$EDUBOT_DIR/tmp"

# Create .env file if it doesn't exist
if [ ! -f "$EDUBOT_DIR/scripts/.env" ]; then
    echo "[INFO] Creating .env file template..."
    cat > "$EDUBOT_DIR/scripts/.env" << EOF
# OpenAI API Key for fallback and image generation
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-api-key-here
EOF
    echo "[WARNING] Please edit scripts/.env and add your OpenAI API key"
fi

echo ""
echo "=========================================="
echo "[SUCCESS] Setup complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Edit scripts/.env and add your OPENAI_API_KEY"
echo "2. Run: bash scripts/start.sh"
echo "=========================================="
