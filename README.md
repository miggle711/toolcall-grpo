# toolcall-grpo

Teach a small open model to call functions correctly using GRPO, with a reward computed by code.

Status: one experiment finished with no clear gain from training alone, a second is running. See Results below.

## Question

Can GRPO with a graded, rule-based reward improve a 1.5B instruct model's function calling on tools it never saw in training, and how much of the gain is real versus reward hacking?

## Plan

1. Baseline: score the untrained model on a held-out split and a Berkeley Function Calling Leaderboard (BFCL) subset.
2. Data: normalise public function-calling datasets into one format (request, tool schemas, gold calls). Split by tool name so the test set contains unseen tools.
3. Reward: graded score for valid JSON, function name, argument match, and extra calls. Unit tested before any training.
4. Train: GRPO with LoRA on one GPU (Hugging Face TRL), checkpoint and resume under Slurm.
5. Evaluate: trained adapter against the baseline on the same scoring code, plus a read-through of failures.

## Layout

| Path | Purpose |
| --- | --- |
| `src/toolcall_grpo/` | data conversion, reward, training and evaluation code |
| `configs/` | one file per run, so every result maps to a config |
| `scripts/` | Slurm job scripts and environment setup |
| `tests/` | unit tests for the reward and data conversion |
| `results/` | small result tables and plots only, no checkpoints |

## Environment

Developed against Monash M3 (Slurm, A40 on the `desktop` partition). Compute nodes have no internet, so models and datasets are downloaded first and jobs run offline (`HF_HUB_OFFLINE=1`). Large files (environments, model cache, checkpoints) live on scratch, not in this repository.

## Running on a Slurm cluster

All paths come from one scratch directory, `BASE`, which holds `envs/`, `hf-cache/`, `data/`, `runs/` and a copy of this repository in `code/`.

```bash
export BASE=/path/to/scratch/toolcall_grpo
bash scripts/setup_env.sh                                  # login node, needs internet
HF_HOME=$BASE/hf-cache python scripts/download_assets.py   # login node, needs internet
python -m toolcall_grpo.data --hermes FILE --out $BASE/data
cd $BASE && sbatch code/scripts/eval_baseline.slurm        # compute node, offline
```

Each run records its config, seed and git commit.

## Results

### First experiment: GRPO from the plain prompt

Qwen2.5-1.5B-Instruct with LoRA (rank 16), 400 GRPO steps of 2 prompts x 8 completions, trained on 937 examples from the Hermes function calling data. Scored on 152 held-out examples whose tool names never appear in training, with greedy decoding. Raw numbers are in `results/first_run/`.

| Run | Mean reward | Exact match | Examples with too many calls |
| --- | --- | --- | --- |
| Untrained | 0.728 | 46.1% | 40 |
| Untrained + hint | 0.733 | 46.1% | 39 |
| Trained | 0.735 | 47.4% | 38 |
| Trained + hint | 0.753 | 48.7% | 33 |

The hint is one sentence appended to each request: "Only call functions whose inputs you already have. If a call needs the result of another call, do not make it yet." The same untrained model scored 0.731 on a different GPU, so the hardware used adds little noise.

Paired comparison on the mean reward, with a 95% bootstrap interval (`python -m toolcall_grpo.compare`):

| Comparison | Change | Interval |
| --- | --- | --- |
| Trained vs untrained | +0.007 | -0.005 to +0.019 |
| Hint alone vs untrained | +0.004 | -0.005 to +0.016 |
| Trained + hint vs untrained | +0.025 | +0.005 to +0.046 |

Reading: training alone did not measurably change the model (135 of 152 examples scored the same). The hint alone changed little on the test set, though on the separate validation split it gave +0.018 (interval +0.003 to +0.037, with 8 of 169 examples changing), so a small effect is possible. The combination of training and hint moved the test set by about 4 more exact answers out of 152. That is weak evidence, because six comparisons were made and the interval only just excludes zero, so it is not a claim of improvement.

### What the baseline gets wrong

Of the 82 failures of the untrained model, 40 call more functions than the gold label. In the cases read by hand, the extra calls were ones the user explicitly asked for, but the gold label keeps only the calls that can be made right now, because later ones depend on earlier results. The dataset therefore encodes a convention the prompt never states, which is what the hint tries to supply. A further 21 failures have gold values that are not literally in the request, some of which are label ambiguity, so exact match has a ceiling below 100%.

### Limits

- 152 test examples and one training run per condition.
- The hint was written after reading the baseline's failures, so it is informed by the data.
- 793 of the 1,893 rows in the source file are dropped because their gold labels put the function name inside the arguments.
- An evaluation of the step-200 checkpoint failed because older checkpoints had been deleted. The second experiment keeps them.

### Second experiment (running)

Train with the hint in every training prompt, 4 prompts per step, on a training file that excludes a validation split. Tuning is judged on the validation split and the test set is looked at once at the end.

Reference numbers on the 169 validation examples, for the untrained model: mean reward 0.772 and exact match 53.3%, and with the hint 0.790 and 55.6% (`results/validation_baselines/`).
