#!/usr/bin/env bash
# ==============================================================================
# Localtunnel Runner for CensorMyPy
# Free, unlimited bandwidth tunnel (no 1GB cap like ngrok)
# ==============================================================================

export PATH="$HOME/.local/bin:$HOME/.local/share/nodejs/bin:$PATH"

PORT=8000
SUBDOMAIN="${1:-censormypy}"

if ! command -v lt >/dev/null 2>&1; then
    echo "Error: localtunnel (lt) not found in PATH."
    exit 1
fi

echo "============================================================"
echo " Starting Localtunnel on Port $PORT (Subdomain: $SUBDOMAIN)"
echo " Fetching Tunnel Password IP..."
TUNNEL_IP=$(curl -s https://loca.lt/mytunnelpassword 2>/dev/null || echo "Unknown")
echo " 🔑 Tunnel Password (if prompted in browser): $TUNNEL_IP"
echo "============================================================"

# Auto-reconnect loop if localtunnel server drops
while true; do
    echo "[$(date '+%H:%M:%S')] Connecting to localtunnel.me..."
    lt --port "$PORT" --subdomain "$SUBDOMAIN"
    echo "[$(date '+%H:%M:%S')] Localtunnel disconnected. Reconnecting in 3 seconds..."
    sleep 3
done
