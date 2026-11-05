"""Score a model on the held-out tool calling examples with the same code the reward uses.

Run with: python -m toolcall_grpo.evaluate --model NAME --data test.jsonl --out DIR
Generation uses vLLM and needs a GPU. score_outputs is pure Python.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from toolcall_grpo.calls import parse_calls
from toolcall_grpo.reward import format_reward, reward_value


# The dataset's gold labels keep only the calls that can be made right now. This sentence states that
# convention, to measure how much of any gain from training a plain instruction would give for free.
HINT = (
    "Only call functions whose inputs you already have. "
    "If a call needs the result of another call, do not make it yet."
)
_END_OF_USER_TURN = "<|im_end|>\n<|im_start|>assistant\n"


def add_hint(prompt: str, hint: str = HINT) -> str:
    """Append the hint to the user's message in a prompt already rendered with the Qwen chat template."""
    if not prompt.endswith(_END_OF_USER_TURN):
        raise ValueError("prompt does not end with the Qwen assistant turn marker")
    return prompt[: -len(_END_OF_USER_TURN)] + "\n\n" + hint + _END_OF_USER_TURN


def score_outputs(examples: list[dict], texts: list[str]) -> tuple[list[dict], dict]:
    rows = []
    for ex, text in zip(examples, texts):
        gold = json.loads(ex["gold_calls"])
        parsed = parse_calls(text)
        predicted = Counter(c["name"] for c in parsed.calls)
        wanted = Counter(c["name"] for c in gold)
        reward = reward_value(text, gold)
        rows.append(
            {
                "id": ex["id"],
                "n_gold": len(gold),
                "reward": reward,
                "exact": abs(reward - 1.0) < 1e-9,
                "well_formed": format_reward([text])[0] == 1.0,
                "name_recall": sum((predicted & wanted).values()) / len(gold),
                "output": text,
            }
        )

    def mean(key, subset=rows):
        return statistics.mean(float(r[key]) for r in subset) if subset else float("nan")

    by_size = {}
    for label, keep in (("1 call", lambda n: n == 1), ("2 calls", lambda n: n == 2), ("3+ calls", lambda n: n >= 3)):
        subset = [r for r in rows if keep(r["n_gold"])]
        by_size[label] = {"n": len(subset), "mean_reward": mean("reward", subset)}
    summary = {
        "n": len(rows),
        "mean_reward": mean("reward"),
        "exact_match": mean("exact"),
        "well_formed": mean("well_formed"),
        "name_recall": mean("name_recall"),
        "by_number_of_gold_calls": by_size,
    }
    return rows, summary


def generate(
    model: str, prompts: list[str], max_new_tokens: int, max_model_len: int, adapter: str | None = None
) -> list[str]:
    from vllm import LLM, SamplingParams

    lora_args = dict(enable_lora=True, max_lora_rank=64) if adapter else {}
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.85, seed=0, **lora_args)
    lora = None
    if adapter:
        from vllm.lora.request import LoRARequest

        lora = LoRARequest("adapter", 1, adapter)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new_tokens), lora_request=lora)
    return [o.outputs[0].text for o in outputs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="base model; read from the adapter when --adapter is given")
    ap.add_argument("--adapter", help="directory of a saved LoRA adapter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--hint", action="store_true", help="append a one sentence instruction to every request")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    model = args.model
    if args.adapter and not model:
        model = json.load(open(Path(args.adapter) / "adapter_config.json"))["base_model_name_or_path"]
    if not model:
        ap.error("give --model, or --adapter to read the base model from it")

    examples = [json.loads(line) for line in open(args.data)][: args.limit]
    prompts = [add_hint(e["prompt"]) if args.hint else e["prompt"] for e in examples]
    texts = generate(model, prompts, args.max_new_tokens, args.max_model_len, args.adapter)
    rows, summary = score_outputs(examples, texts)
    summary["model"] = model
    summary["adapter"] = args.adapter
    summary["hint"] = HINT if args.hint else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
