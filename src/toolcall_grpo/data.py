"""Turn Hermes function calling data into GRPO prompts, split so test tools are never seen in training.

Run with: python -m toolcall_grpo.data --hermes FILE --tokenizer NAME --out DIR
"""

import argparse
import hashlib
import json
from pathlib import Path

from toolcall_grpo.calls import parse_calls


def convert_row(row: dict, render_prompt) -> dict | None:
    """One Hermes row to one example, or None when its gold calls are unusable.

    The dataset's own system prompt is dropped: the chat template of the model being trained
    supplies its native tool instructions from the tool schemas.
    """
    turns = {c["from"]: c["value"] for c in row["conversations"]}
    gold = parse_calls(turns.get("gpt", ""), lenient=True)
    if not gold.calls or len(gold.calls) != gold.n_blocks:
        return None
    tools = json.loads(row["tools"])
    offered = {t["function"]["name"] for t in tools}
    if any(c["name"] not in offered for c in gold.calls):
        return None
    return {
        "id": row["id"],
        "category": row["category"],
        "prompt": render_prompt([{"role": "user", "content": turns["human"]}], tools),
        "gold_calls": json.dumps(gold.calls),  # a string: Arrow cannot hold ragged dicts
        "tool_names": sorted({c["name"] for c in gold.calls}),
    }


def split_by_tool_group(examples: list[dict], test_fraction: float, seed: int):
    """Examples that share any tool name stay together, so no test tool name appears in training."""
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ex in examples:
        first, *rest = ex["tool_names"]
        for name in rest:
            parent[find(name)] = find(first)
        find(first)

    def is_test(ex: dict) -> bool:
        key = f"{seed}:{find(ex['tool_names'][0])}".encode()
        return int(hashlib.sha256(key).hexdigest(), 16) % 10_000 < test_fraction * 10_000

    train = [ex for ex in examples if not is_test(ex)]
    test = [ex for ex in examples if is_test(ex)]
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hermes", required=True, help="func-calling-singleturn.json")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--val-fraction", type=float, default=0.15, help="share of train held out for tuning")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def render(messages, tools):
        return tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)

    rows = json.load(open(args.hermes))
    converted = [convert_row(r, render) for r in rows]
    usable = [e for e in converted if e is not None]
    for e in usable:
        e["prompt_tokens"] = len(tok(e["prompt"]).input_ids)
    kept = [e for e in usable if e["prompt_tokens"] <= args.max_prompt_tokens]
    train, test = split_by_tool_group(kept, args.test_fraction, args.seed)
    # Tune on val and look at test once at the end. val is carved out of train, so train.jsonl and
    # test.jsonl stay exactly as they were. train_fit.jsonl is train without val.
    train_fit, val = split_by_tool_group(train, args.val_fraction, args.seed + 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("test", test), ("train_fit", train_fit), ("val", val)):
        with open(out / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in part)
    print(
        f"rows={len(rows)} usable={len(usable)} within {args.max_prompt_tokens} tokens={len(kept)} "
        f"train={len(train)} test={len(test)} (train_fit={len(train_fit)} val={len(val)})"
    )


if __name__ == "__main__":
    main()
