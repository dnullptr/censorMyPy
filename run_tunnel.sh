#!/usr/bin/env bash
# Quick helper to start a Cloudflare tunnel for CensorMyPy
CLOUDFLARED="${HOME}/.local/bin/cloudflared"

if [ ! -f "$CLOUDFLARED" ]; then
    if command -v cloudflared >/dev/null 2>&1; then
        CLOUDFLARED="cloudflared"
    else
        echo "Error: cloudflared not found in ~/.local/bin or PATH."
        exit 1
    fi
fi

echo "========================================================="
echo " Starting Cloudflare Quick Tunnel -> http://localhost:8000"
echo " A public https://*.trycloudflare.com URL will appear below"
echo "========================================================="
exec "$CLOUDFLARED" tunnel --url http://localhost:8000
