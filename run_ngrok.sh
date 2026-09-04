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

DOMAIN="${1:-$NGROK_DOMAIN}"

if [ -n "$DOMAIN" ]; then
    # Strip leading protocol if present
    CLEAN_DOMAIN=$(echo "$DOMAIN" | sed -e 's|^https://||' -e 's|^http://||')
    echo "================================================================="
    echo " Starting ngrok with static domain: https://$CLEAN_DOMAIN"
    echo " Forwarding to local Gradio server on http://localhost:8000"
    echo "================================================================="
    exec "$NGROK" http --url="https://$CLEAN_DOMAIN" 8000
else
    echo "================================================================="
    echo " Starting ngrok -> http://localhost:8000"
    echo " Tip: Run with your domain: ./run_ngrok.sh your-name.ngrok-free.app"
    echo "================================================================="
    exec "$NGROK" http 8000
fi
