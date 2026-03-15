#!/bin/bash
# Set working directory
export NVM_DIR="/home/pi/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Find the sound card index for wm8960soundcard
card_index=$(awk '/wm8960soundcard/ {print $1}' /proc/asound/cards | head -n1)
# Default to 1 if not found
if [ -z "$card_index" ]; then
  card_index=1
fi
echo "Using sound card index: $card_index"

# Output current environment information (for debugging)
echo "===== Start time: $(date) =====" 
echo "Current user: $(whoami)" 
echo "Working directory: $(pwd)" 
working_dir=$(pwd)
echo "PATH: $PATH" 
echo "Python version: $(python3 --version)" 
echo "Node version: $(node --version)"
sleep 5

# Start the service
echo "Starting Node.js application..."
cd $working_dir

get_env_value() {
  if grep -Eq "^[[:space:]]*$1[[:space:]]*=" .env; then
    val=$(grep -E "^[[:space:]]*$1[[:space:]]*=" .env | tail -n1 | cut -d'=' -f2-)
    # trim whitespace and surrounding quotes
    echo "$(echo "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
  else
    echo ""
  fi
}

# load .env variables, exclude comments and empty lines
# check if .env file exists
initial_volume_level=114
serve_ollama=false
if [ -f ".env" ]; then
  # Load only SERVE_OLLAMA from .env (ignore comments/other vars)
  SERVE_OLLAMA=$(get_env_value "SERVE_OLLAMA")
  [ -n "$SERVE_OLLAMA" ] && export SERVE_OLLAMA
  
  CUSTOM_FONT_PATH=$(get_env_value "CUSTOM_FONT_PATH")
  [ -n "$CUSTOM_FONT_PATH" ] && export CUSTOM_FONT_PATH

  INITIAL_VOLUME_LEVEL=$(get_env_value "INITIAL_VOLUME_LEVEL")
  [ -n "$INITIAL_VOLUME_LEVEL" ] && export INITIAL_VOLUME_LEVEL

  WHISPER_MODEL_SIZE=$(get_env_value "WHISPER_MODEL_SIZE")
  [ -n "$WHISPER_MODEL_SIZE" ] && export WHISPER_MODEL_SIZE

  FASTER_WHISPER_MODEL_SIZE=$(get_env_value "FASTER_WHISPER_MODEL_SIZE")
  [ -n "$FASTER_WHISPER_MODEL_SIZE" ] && export FASTER_WHISPER_MODEL_SIZE

  echo ".env variables loaded."

  # check if SERVE_OLLAMA is set to true
  if [ "$SERVE_OLLAMA" = "true" ]; then
    serve_ollama=true
  fi

  if [ -n "$INITIAL_VOLUME_LEVEL" ]; then
    initial_volume_level=$INITIAL_VOLUME_LEVEL
  fi
else
  echo ".env file not found, please create one based on .env.template."
  exit 1
fi

# Adjust initial volume
amixer -c $card_index set Speaker $initial_volume_level

if [ "$serve_ollama" = true ]; then
  echo "Starting Ollama server..."
  export OLLAMA_KEEP_ALIVE=-1 # ensure Ollama server stays alive
  OLLAMA_HOST=0.0.0.0:11434 ollama serve &
fi

# ── O.A.S.I.S. RAG Service ───────────────────────────────────────────────────
echo "Starting O.A.S.I.S. RAG service..."
OASIS_RAG_LOG="$working_dir/oasis-rag.log"
OASIS_RAG_SERVICE_URL_VAL=$(get_env_value "OASIS_RAG_SERVICE_URL")
OASIS_RAG_SERVICE_URL_VAL="${OASIS_RAG_SERVICE_URL_VAL:-http://localhost:5001}"
RAG_HEALTH_URL="${OASIS_RAG_SERVICE_URL_VAL}/health"

python3 "$working_dir/python/oasis-rag/service.py" \
  >> "$OASIS_RAG_LOG" 2>&1 &
RAG_PID=$!
echo "RAG service started (PID=$RAG_PID). Log: $OASIS_RAG_LOG"

# Wait up to 30 s for the service to become healthy (3 s intervals × 10 tries)
rag_ready=false
for i in $(seq 1 10); do
  sleep 3
  health=$(curl -sf --max-time 2 "$RAG_HEALTH_URL" 2>/dev/null)
  if echo "$health" | grep -q '"status":"ok"'; then
    rag_ready=true
    echo "RAG service ready (${i}x3s elapsed)."
    break
  fi
  echo "Waiting for RAG service... ($((i * 3))s / 30s)"
done

if [ "$rag_ready" = false ]; then
  echo "WARNING: RAG service did not become ready within 30s."
  echo "  Chatbot will start with fallback protocol matching."
  echo "  Check $OASIS_RAG_LOG for details."
fi
# ─────────────────────────────────────────────────────────────────────────────

# if file use_npm exists and is true, use npm
if [ -f "use_npm" ]; then
  use_npm=true
else
  use_npm=false
fi

if [ "$use_npm" = true ]; then
  echo "Using npm to start the application..."
  SOUND_CARD_INDEX=$card_index npm start
else
  echo "Using yarn to start the application..."
  SOUND_CARD_INDEX=$card_index yarn start
fi

# After the service ends, perform cleanup
echo "Cleaning up after service..."

if [ -n "$RAG_PID" ] && kill -0 "$RAG_PID" 2>/dev/null; then
  echo "Stopping RAG service (PID=$RAG_PID)..."
  kill "$RAG_PID"
fi

if [ "$serve_ollama" = true ]; then
  echo "Stopping Ollama server..."
  pkill ollama
fi

# Record end status
echo "===== Service ended: $(date) ====="
