#!/bin/bash
# Score a model or a LoRA adapter on the held-out examples. Compute nodes have no internet.
# Submit from the project scratch directory. Arguments go to toolcall_grpo.evaluate, with paths
# relative to that directory:
#   cd $BASE && sbatch code/scripts/evaluate.slurm --adapter runs/grpo_1p5b/final --out results/grpo_1p5b
#SBATCH --job-name=tc-eval
#SBATCH --partition=desktop
#SBATCH --qos=desktopq
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
BASE=${BASE:-$SLURM_SUBMIT_DIR}
cd "$BASE"

export HF_HOME=$BASE/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=$BASE/cache/vllm TORCHINDUCTOR_CACHE_DIR=$BASE/cache/inductor
export TRITON_CACHE_DIR=$BASE/cache/triton XDG_CACHE_HOME=$BASE/cache/xdg TMPDIR=$BASE/tmp
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# Compute nodes have no nvcc, so FlashInfer cannot compile its sampling kernel at runtime.
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=$BASE/cache/flashinfer
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

source "$BASE/envs/trl/bin/activate"
export PYTHONPATH=$BASE/code/src

echo "== code version"
cat "$BASE/code/GIT_COMMIT" 2>/dev/null || echo unknown
python -m toolcall_grpo.evaluate --data "$BASE/data/test.jsonl" "$@"
