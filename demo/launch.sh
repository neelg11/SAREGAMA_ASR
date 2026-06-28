#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch.sh — start the Hindi Singing ASR server.
#
# Usage:
#   ./launch.sh                 # use defaults
#   MODEL_PATH=/path/to/ckpt ./launch.sh
#   PORT=8080 ./launch.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve to this script's directory so it works from anywhere.
cd "$(dirname "$0")"

export MODEL_PATH="${MODEL_PATH:-whisper-large-v3-turbo-merged}"
export IDLE_TIMEOUT_SECONDS="${IDLE_TIMEOUT_SECONDS:-300}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-7860}"

echo "──────────────────────────────────────────────"
echo "  Hindi Singing ASR"
echo "  model path : ${MODEL_PATH}"
echo "  idle sleep : ${IDLE_TIMEOUT_SECONDS}s"
echo "  serving on : http://${HOST}:${PORT}"
echo "──────────────────────────────────────────────"

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "WARNING: model path '${MODEL_PATH}' not found."
  echo "Set MODEL_PATH to your merged checkpoint, e.g.:"
  echo "    MODEL_PATH=/data/whisper-large-v3-turbo-merged ./launch.sh"
fi

exec python app.py
