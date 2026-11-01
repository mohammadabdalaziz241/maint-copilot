#!/usr/bin/env bash
# One-time setup on the Surrey machine. Everything (venv, caches, model weights) goes to scratch:
# the home folder has only ~5 GB free.
set -euo pipefail
BASE=${MC_BASE:-/scratch/$USER/maint-copilot-cluster}
mkdir -p "$BASE"/{cache/uv,cache/pip,cache/hf,logs,results}
export UV_CACHE_DIR=$BASE/cache/uv PIP_CACHE_DIR=$BASE/cache/pip HF_HOME=$BASE/cache/hf
export XDG_CACHE_HOME=$BASE/cache TMPDIR=$BASE/cache

if [ ! -d "$BASE/venv" ]; then
  python3 -m venv "$BASE/venv"
fi
# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
pip install -q -U pip uv
# torch comes from PyPI (CUDA build); download.pytorch.org is blocked from this machine.
uv pip install vllm openai pydantic pandas pyarrow

python - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU", p.name, f"{p.total_memory / 2**30:.1f} GiB", "compute capability", f"{p.major}.{p.minor}")
PY
df -h "$BASE" | tail -1
