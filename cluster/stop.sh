#!/usr/bin/env bash
BASE=${MC_BASE:-/scratch/$USER/maint-copilot-cluster}
if [ -f "$BASE/logs/serve.pid" ]; then kill "$(cat "$BASE/logs/serve.pid")" 2>/dev/null && echo stopped; rm -f "$BASE/logs/serve.pid"; fi
