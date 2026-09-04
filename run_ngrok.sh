#!/usr/bin/env bash
set -e

NGROK="${HOME}/.local/bin/ngrok"

if [ ! -f "$NGROK" ]; then
    if command -v ngrok >/dev/null 2>&1; then
        NGROK="ngrok"
    else
        echo "Error: ngrok binary not found."
        exit 1
    fi
fi

DEFAULT_DOMAIN="ridden-ammonia-eternity.ngrok-free.dev"
DOMAIN="${1:-${NGROK_DOMAIN:-$DEFAULT_DOMAIN}}"

# Strip leading protocol if present
CLEAN_DOMAIN=$(echo "$DOMAIN" | sed -e 's|^https://||' -e 's|^http://||')

echo "================================================================="
echo " Starting ngrok with static domain: https://$CLEAN_DOMAIN"
echo " Forwarding to local Gradio server on http://localhost:8000"
echo "================================================================="

exec "$NGROK" http --url="https://$CLEAN_DOMAIN" 8000
