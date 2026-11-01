#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server for one model and wait until it is ready.
# Usage: bash serve.sh <hf-model-id> [extra vllm serve args]
set -euo pipefail
MODEL=${1:?model id}
shift || true
BASE=/scratch/ma06314/maint-copilot-cluster
export HF_HOME=$BASE/cache/hf XDG_CACHE_HOME=$BASE/cache TMPDIR=$BASE/cache
# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
SLUG=$(echo "$MODEL" | tr '/' '_')
LOG=$BASE/logs/serve_$SLUG.log
if curl -s -o /dev/null localhost:8000/health; then
  echo "A server is already running on port 8000; stop it first: bash $BASE/cluster/stop.sh"; exit 1
fi
nohup vllm serve "$MODEL" --served-model-name local --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.92 "$@" > "$LOG" 2>&1 &
echo $! > "$BASE/logs/serve.pid"
echo "starting $MODEL (pid $(cat "$BASE/logs/serve.pid")); log: $LOG"
for i in $(seq 1 180); do  # up to 30 min, includes the first download
  if curl -s -o /dev/null localhost:8000/health; then
    echo "ready after ~$((i * 10)) s"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  if ! kill -0 "$(cat "$BASE/logs/serve.pid")" 2>/dev/null; then
    echo "server exited; last log lines:"; tail -40 "$LOG"; exit 1
  fi
  sleep 10
done
echo "not ready after 30 min; see $LOG"; exit 1
