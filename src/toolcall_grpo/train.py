"""GRPO with LoRA on tool calling.

Run with: python -m toolcall_grpo.train --config configs/smoke.yaml --output-dir DIR --data-dir DIR

Resumes from the newest checkpoint in the output directory, so a requeued Slurm job continues.
"""

import argparse
import json
from pathlib import Path

import yaml


def load_examples(path: Path, limit: int | None, hint: bool = False) -> list[dict]:
    from toolcall_grpo.evaluate import add_hint

    rows = [json.loads(line) for line in open(path)][:limit]
    # Only these columns reach the reward function, gold_calls as a JSON string.
    return [{"prompt": add_hint(r["prompt"]) if hint else r["prompt"], "gold_calls": r["gold_calls"]} for r in rows]


def build_configs(cfg: dict, output_dir: Path):
    """The trainer and LoRA configs from a parsed YAML file. Shared with the tests."""
    from peft import LoraConfig
    from trl import GRPOConfig

    return (
        GRPOConfig(output_dir=str(output_dir), **cfg["grpo"]),
        LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
    )


def silence_completion_table() -> None:
    # The console table prints every full prompt (about 700 tokens of tool schemas) for each logged
    # step. The same completions are still saved as parquet files in the output directory.
    import trl.trainer.grpo_trainer as grpo_trainer

    grpo_trainer.print_prompt_completions_sample = lambda *args, **kwargs: None


def report_peak_gpu_memory() -> None:
    import torch

    if torch.cuda.is_available():
        gib = 2**30
        total = torch.cuda.get_device_properties(0).total_memory / gib
        print(
            f"peak GPU memory: allocated {torch.cuda.max_memory_allocated() / gib:.1f} GiB, "
            f"reserved {torch.cuda.max_memory_reserved() / gib:.1f} GiB, device total {total:.1f} GiB"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override grpo.max_steps, for short probe runs")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOTrainer

    from toolcall_grpo.reward import call_reward, format_reward

    cfg = yaml.safe_load(open(args.config))
    if args.max_steps is not None:
        cfg["grpo"]["max_steps"] = args.max_steps
    if not cfg.get("print_completions", True):
        silence_completion_table()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = Dataset.from_list(
        load_examples(
            Path(args.data_dir) / cfg.get("train_file", "train.jsonl"),
            cfg.get("max_train_examples"),
            hint=cfg.get("train_hint", False),
        )
    )
    grpo, lora = build_configs(cfg, out)
    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=[format_reward, call_reward],
        args=grpo,
        train_dataset=train,
        peft_config=lora,
    )
    resume = get_last_checkpoint(str(out))
    print(f"train examples={len(train)} resume_from={resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out / "final"))
    report_peak_gpu_memory()


if __name__ == "__main__":
    main()
