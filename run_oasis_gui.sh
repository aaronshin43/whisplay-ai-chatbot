#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Load .env
set -a
[ -f .env ] && source .env
set +a

# Sound card detection (Pi)
card_index=$(awk '/wm8960soundcard/ {print $1}' /proc/asound/cards 2>/dev/null | head -n1)
card_index=${card_index:-${SOUND_CARD_INDEX:-1}}
echo "Using sound card index: $card_index"
amixer -c "$card_index" set Speaker "${INITIAL_VOLUME_LEVEL:-114}" 2>/dev/null || true
export SOUND_CARD_INDEX="$card_index"

# Ollama (optional)
OLLAMA_PID=""
if [ "${SERVE_OLLAMA:-false}" = "true" ]; then
    echo "Starting Ollama server..."
    export OLLAMA_KEEP_ALIVE=-1
    OLLAMA_HOST=0.0.0.0:11434 ollama serve > /tmp/oasis_ollama.log 2>&1 &
    OLLAMA_PID=$!
    sleep 2
fi

# Classify service (:5002)
echo "Starting classify service..."
python3 python/oasis-classify/service.py > /tmp/oasis_classify.log 2>&1 &
CLASSIFY_PID=$!

# Faster-whisper ASR (:8803)
ASR_LOG=/tmp/oasis_asr.log
echo "Starting ASR service... (log: $ASR_LOG)"
python3 python/speech-service/faster-whisper-host.py --port "${FASTER_WHISPER_PORT:-8803}" > "$ASR_LOG" 2>&1 &
ASR_PID=$!

# Wait for classify service health (up to 30s)
echo "Waiting for classify service..."
for i in $(seq 1 10); do
    if curl -sf http://127.0.0.1:5002/health >/dev/null 2>&1; then
        echo "Classify service ready (${i}x3s elapsed)."
        break
    fi
    sleep 3
done

# Wait for ASR service health (up to 60s — whisper model load takes ~10s on Pi)
echo "Waiting for ASR service..."
asr_ready=false
for i in $(seq 1 20); do
    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
        -X POST http://127.0.0.1:"${FASTER_WHISPER_PORT:-8803}"/recognize \
        -H "Content-Type: application/json" -d '{}' 2>/dev/null || echo "000")
    if [ "$status" = "400" ] || [ "$status" = "200" ]; then
        echo "ASR service ready (${i}x3s elapsed)."
        asr_ready=true
        break
    fi
    sleep 3
done

if [ "$asr_ready" = "false" ]; then
    echo "WARNING: ASR service did not start. Check $ASR_LOG"
    tail -20 "$ASR_LOG"
fi

# GUI (main entry point)
echo "Starting oasis-gui..."
python3 python/oasis-gui/main.py
EXIT_CODE=$?

# Cleanup background services
echo "Shutting down services..."
kill "$CLASSIFY_PID" "$ASR_PID" 2>/dev/null || true
if [ -n "$OLLAMA_PID" ]; then kill "$OLLAMA_PID" 2>/dev/null || true; fi

exit $EXIT_CODE
