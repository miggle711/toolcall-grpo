#!/usr/bin/env bash
# Build the Python environment on Monash M3 scratch.
# Run on the login node: compute nodes have no internet.
# Home has very little free space, so every cache and install location is on scratch.
set -euo pipefail

BASE=${BASE:?set BASE to a scratch directory with plenty of space, not your home directory}
mkdir -p "$BASE"/{envs,hf-cache,uv-cache,uv-python,data,runs,tmp}

export UV_CACHE_DIR="$BASE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$BASE/uv-python"
export TMPDIR="$BASE/tmp"

# System python is 3.9, TRL needs 3.11 or newer.
uv venv --python 3.12 "$BASE/envs/trl"
# shellcheck disable=SC1091
source "$BASE/envs/trl/bin/activate"

# trl[vllm] pins vllm below 0.33.0 and pulls the torch build that vllm needs.
uv pip install "trl[vllm]" peft accelerate datasets "huggingface_hub[cli]" tensorboard pytest

python - <<'PY'
import importlib.metadata as md, sys
print("python", sys.version.split()[0])
for p in ["torch", "trl", "vllm", "transformers", "peft", "accelerate", "datasets"]:
    print(f"{p:14s}", md.version(p))
PY
du -sh "$BASE/envs/trl" "$BASE/uv-cache" "$BASE/uv-python"
