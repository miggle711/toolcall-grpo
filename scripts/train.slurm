#!/bin/bash
# GRPO training on one A40 (desktopq allows one GPU per user). Compute nodes have no internet.
# Submit from the project scratch directory, giving a config. Anything after the config goes to the trainer.
#   cd $BASE && sbatch --time=00:45:00 code/scripts/train.slurm configs/smoke.yaml
# The SFT baseline uses the same script: MODULE=toolcall_grpo.train_sft RUN_NAME=sft_1p5b sbatch --export=ALL code/scripts/train.slurm configs/sft/sft_1p5b.yaml
# A short probe of a real config, in its own output directory so it cannot be mistaken for the real run:
#   cd $BASE && RUN_NAME=probe sbatch --export=ALL --time=00:30:00 code/scripts/train.slurm configs/grpo_1p5b.yaml --max-steps 20
#SBATCH --job-name=tc-train
#SBATCH --partition=desktop
#SBATCH --qos=desktopq
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
CONFIG=${1:?give a config path relative to code/, for example configs/smoke.yaml}
shift
BASE=${BASE:-$SLURM_SUBMIT_DIR}
# The output directory is named after the run, so a requeued job resumes from its own checkpoints.
RUN_NAME=${RUN_NAME:-$(basename "$CONFIG" .yaml)}
OUT=$BASE/runs/$RUN_NAME
mkdir -p "$OUT"

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
echo "== code version"
cat "$BASE/code/GIT_COMMIT" 2>/dev/null || echo unknown
echo "== config $CONFIG"
cat "$BASE/code/$CONFIG"

# Record device memory every 10 seconds, and report the peak when the job ends, however it ends.
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 10 > "$OUT/gpu_memory.csv" &
SAMPLER=$!
report_peak() {
    kill "$SAMPLER" 2>/dev/null || true
    echo "== peak GPU memory used (MiB): $(awk -F', ' '$2 > m { m = $2 } END { print m + 0 }' "$OUT/gpu_memory.csv")"
}
trap report_peak EXIT

python -m "${MODULE:-toolcall_grpo.train}" \
    --config "$BASE/code/$CONFIG" \
    --data-dir "$BASE/data" \
    --output-dir "$OUT" \
    "$@"
