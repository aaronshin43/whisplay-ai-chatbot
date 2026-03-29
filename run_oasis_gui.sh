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
    OLLAMA_HOST=0.0.0.0:11434 ollama serve &
    OLLAMA_PID=$!
    sleep 2
fi

# Classify service (:5002)
echo "Starting classify service..."
python3 python/oasis-classify/service.py &
CLASSIFY_PID=$!

# Faster-whisper ASR (:8803)
echo "Starting ASR service..."
python3 python/speech-service/faster-whisper-host.py --port "${FASTER_WHISPER_PORT:-8803}" &
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

# GUI (main entry point)
echo "Starting oasis-gui..."
python3 python/oasis-gui/main.py
EXIT_CODE=$?

# Cleanup background services
echo "Shutting down services..."
kill "$CLASSIFY_PID" "$ASR_PID" 2>/dev/null || true
if [ -n "$OLLAMA_PID" ]; then kill "$OLLAMA_PID" 2>/dev/null || true; fi

exit $EXIT_CODE
