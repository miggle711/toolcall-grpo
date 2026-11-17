"""Supervised finetuning with LoRA on the gold tool calls, as a baseline for the GRPO runs.

Run with: python -m toolcall_grpo.train_sft --config configs/sft/sft_1p5b.yaml --output-dir DIR --data-dir DIR

It trains on the same examples as GRPO, so the two methods are compared on equal data. The loss is
computed on the completion only, not on the tool schemas in the prompt.
"""

import argparse
import json
from pathlib import Path

import yaml


def render_completion(gold_calls: str) -> str:
    """The gold calls as the model must write them: one <tool_call> block per call."""
    return "\n".join(f"<tool_call>\n{json.dumps(c)}\n</tool_call>" for c in json.loads(gold_calls))


def load_sft_examples(path: Path, limit: int | None, hint: bool = False) -> list[dict]:
    from toolcall_grpo.evaluate import add_hint

    rows = [json.loads(line) for line in open(path)][:limit]
    return [
        {
            "prompt": add_hint(r["prompt"]) if hint else r["prompt"],
            "completion": render_completion(r["gold_calls"]),
        }
        for r in rows
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override sft.max_steps, for short probe runs")
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    from toolcall_grpo.train import report_peak_gpu_memory

    cfg = yaml.safe_load(open(args.config))
    if args.max_steps is not None:
        cfg["sft"]["max_steps"] = args.max_steps
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = Dataset.from_list(
        load_sft_examples(
            Path(args.data_dir) / cfg.get("train_file", "train_fit.jsonl"),
            cfg.get("max_train_examples"),
            hint=cfg.get("train_hint", False),
        )
    )
    trainer = SFTTrainer(
        model=cfg["model"],
        args=SFTConfig(output_dir=str(out), **cfg["sft"]),
        train_dataset=train,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
    )
    resume = get_last_checkpoint(str(out))
    print(f"train examples={len(train)} resume_from={resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out / "final"))
    report_peak_gpu_memory()


if __name__ == "__main__":
    main()
