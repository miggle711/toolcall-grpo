#!/bin/bash
# Score the untrained models on the held-out tool calling examples.
# One A40 on the desktop partition (desktopq allows one GPU per user). Compute nodes have no internet.
#SBATCH --job-name=tc-baseline
#SBATCH --partition=desktop
#SBATCH --qos=desktopq
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=runs/%x-%j.out

# Submit from the project scratch directory, which holds code/, data/, runs/ and envs/:
#   cd $BASE && sbatch code/scripts/eval_baseline.slurm
set -euo pipefail
BASE=${BASE:-$SLURM_SUBMIT_DIR}

export HF_HOME=$BASE/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=$BASE/cache/vllm TORCHINDUCTOR_CACHE_DIR=$BASE/cache/inductor
export TRITON_CACHE_DIR=$BASE/cache/triton XDG_CACHE_HOME=$BASE/cache/xdg TMPDIR=$BASE/tmp
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# Compute nodes have no nvcc, so FlashInfer cannot compile its sampling kernel at runtime.
# Use vLLM's own sampler, and keep any FlashInfer cache off the small home directory.
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=$BASE/cache/flashinfer
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

source "$BASE/envs/trl/bin/activate"
export PYTHONPATH=$BASE/code/src

echo "== node and GPU"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "== code version"
cat "$BASE/code/GIT_COMMIT" 2>/dev/null || echo unknown

for MODEL in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct; do
    echo "== evaluating $MODEL"
    python -m toolcall_grpo.evaluate \
        --model "$MODEL" \
        --data "$BASE/data/test.jsonl" \
        --out "$BASE/results/baseline/$(basename "$MODEL")"
done
