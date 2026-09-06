#!/usr/bin/env bash
# ==============================================================================
# CensorMyPy Watchdog & Auto-Restart Supervisor
#
# Keeps CensorMyPy Gradio Web UI alive 24/7.
# - Automatically restarts if the process crashes due to OOM / RAM exhaustion.
# - Performs periodic health checks to detect frozen/deadlocked states.
# - Cleans up dangling ports (8000) and temporary files between runs.
# - Logs restart events with timestamps to watchdog.log and console.
# - Stops cleanly when receiving SIGINT (Ctrl+C) or SIGTERM.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/watchdog.log"
PORT=8000
HEALTH_INTERVAL=20     # Seconds between health checks
MAX_FAILED_CHECKS=3    # Consecutive failures before force-killing a hung process
RESTART_DELAY=3        # Seconds to wait before restarting to let OS reclaim RAM

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

cleanup_env() {
    # Free port 8000
    fuser -k ${PORT}/tcp 2>/dev/null || /usr/sbin/fuser -k ${PORT}/tcp 2>/dev/null || true
    pkill -f "gradio_app.py" 2>/dev/null || true
    # Clean temporary scratch files and separated stems
    rm -rf /tmp/cmypy_* 2>/dev/null || true
    rm -f down_temp.wav down_temp.mp3 temp.wav temp.mp3 temp_ts_in.wav temp_ts_down.wav 2>/dev/null || true
    rm -rf separated 2>/dev/null || true
}

SHUTDOWN_REQUESTED=0

handle_stop() {
    log "Received shutdown signal (Ctrl+C / SIGTERM). Stopping CensorMyPy cleanly..."
    SHUTDOWN_REQUESTED=1
    if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
        kill "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    cleanup_env
    log "Watchdog stopped."
    exit 0
}

trap handle_stop SIGINT SIGTERM SIGHUP

log "============================================================"
log " Starting CensorMyPy Watchdog Supervisor"
log " Target: ./run_gradio.sh $*"
log " Log: $LOG_FILE"
log "============================================================"

RESTART_COUNT=0

while [ "$SHUTDOWN_REQUESTED" -eq 0 ]; do
    cleanup_env
    sleep 1

    log "Launching CensorMyPy (Run #$((RESTART_COUNT + 1)))..."
    
    # Run in background and save PID
    ./run_gradio.sh "$@" &
    APP_PID=$!
    
    FAILED_CHECKS=0
    # Allow 10 seconds for initial server startup before checking health
    sleep 10

    # Monitor loop while child process is alive
    while kill -0 "$APP_PID" 2>/dev/null; do
        # Check HTTP health
        if curl -s -f -m 5 "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
            FAILED_CHECKS=0
        else
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            if [ "$FAILED_CHECKS" -ge "$MAX_FAILED_CHECKS" ]; then
                log "WARNING: Server at http://127.0.0.1:${PORT}/ failed $FAILED_CHECKS consecutive health checks (frozen/deadlocked)."
                log "Force-killing frozen process (PID $APP_PID)..."
                kill -9 "$APP_PID" 2>/dev/null || true
                break
            fi
        fi
        
        sleep "$HEALTH_INTERVAL"
    done

    # Wait for exit status
    wait "$APP_PID" 2>/dev/null
    EXIT_CODE=$?

    if [ "$SHUTDOWN_REQUESTED" -eq 1 ]; then
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    log "CRASH / EXIT DETECTED: Process PID $APP_PID exited with code $EXIT_CODE."
    log "Performing cleanup and cooling down for ${RESTART_DELAY}s before auto-restarting..."
    
    cleanup_env
    sleep "$RESTART_DELAY"
done

cleanup_env
log "Watchdog exited."
