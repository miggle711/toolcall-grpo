"""Paired comparison of two evaluation runs on the same examples.

With a small test set, a difference in means is easy to over read. This reports the mean change in
reward with a paired bootstrap interval, and how many examples got better, worse or stayed the same.

Run with: python -m toolcall_grpo.compare --before DIR --after DIR
"""

import argparse
import json
import random
from pathlib import Path


def load(run_dir: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, open(run_dir / "outputs.jsonl"))}


def compare(before: dict[str, dict], after: dict[str, dict], resamples: int = 10_000, seed: int = 0) -> dict:
    ids = sorted(set(before) & set(after))
    if not ids:
        raise ValueError("the two runs share no examples")
    diffs = [after[i]["reward"] - before[i]["reward"] for i in ids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(resamples))
    return {
        "n": len(ids),
        "mean_reward_before": sum(before[i]["reward"] for i in ids) / len(ids),
        "mean_reward_after": sum(after[i]["reward"] for i in ids) / len(ids),
        "mean_change": sum(diffs) / len(diffs),
        "ci95": (means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]),
        "exact_before": sum(before[i]["exact"] for i in ids) / len(ids),
        "exact_after": sum(after[i]["exact"] for i in ids) / len(ids),
        "better": sum(d > 1e-9 for d in diffs),
        "worse": sum(d < -1e-9 for d in diffs),
        "same": sum(abs(d) <= 1e-9 for d in diffs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()
    result = compare(load(Path(args.before)), load(Path(args.after)))
    lo, hi = result["ci95"]
    print(f"examples: {result['n']}")
    print(f"mean reward: {result['mean_reward_before']:.3f} -> {result['mean_reward_after']:.3f}")
    print(f"mean change: {result['mean_change']:+.3f}  (95% bootstrap interval {lo:+.3f} to {hi:+.3f})")
    print(f"exact match: {result['exact_before']:.3f} -> {result['exact_after']:.3f}")
    print(f"better {result['better']}, worse {result['worse']}, same {result['same']}")


if __name__ == "__main__":
    main()
