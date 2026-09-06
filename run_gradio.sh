#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Virtual environment not found at $SCRIPT_DIR/.venv"
    exit 1
fi

# Automatically free port 8000 if previously occupied
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || /usr/sbin/fuser -k 8000/tcp 2>/dev/null || true
pkill -f "gradio_app.py" 2>/dev/null || true
sleep 0.5

echo "===================================================="
echo " Starting CensorMyPy Gradio Web UI on Bazzite (BC-250)"
echo " GPU Engine: AMD BC-250 Vulkan GPU (RADV GFX1013)"
echo " UI will be available at: http://localhost:8000"
echo "===================================================="

export PYTHONUNBUFFERED=1
exec "$SCRIPT_DIR/.venv/bin/python" gradio_app.py "$@"
