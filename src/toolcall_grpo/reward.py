"""Graded rewards for tool calling.

A call earns NAME_CREDIT for the right function name, plus the rest in proportion to an F1 score
over its (argument, value) pairs. Gold calls are matched to predicted calls by name, so parallel
calls may come in any order. Extra blocks beyond the number of gold calls scale the score down.
"""

import json

from toolcall_grpo.calls import parse_calls

NAME_CREDIT = 0.3


def _canon(value) -> str:
    """Canonical text for a value, so 1 and 1.0 match but True and 1 do not."""

    def norm(v):
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {str(k): norm(x) for k, x in v.items()}
        return str(v)

    return json.dumps(norm(value), sort_keys=True)


def arg_f1(pred: dict, gold: dict) -> float:
    if not pred and not gold:
        return 1.0
    hits = sum(1 for k, v in gold.items() if k in pred and _canon(pred[k]) == _canon(v))
    if hits == 0:
        return 0.0
    precision, recall = hits / len(pred), hits / len(gold)
    return 2 * precision * recall / (precision + recall)


def call_score(pred: dict, gold: dict) -> float:
    if pred["name"] != gold["name"]:
        return 0.0
    return NAME_CREDIT + (1 - NAME_CREDIT) * arg_f1(pred["arguments"], gold["arguments"])


def reward_value(text: str, gold_calls: list[dict]) -> float:
    parsed = parse_calls(text)
    if parsed.n_blocks == 0 or not gold_calls:
        return 0.0
    unused = list(parsed.calls)
    total = 0.0
    for gold in gold_calls:
        scored = [(call_score(p, gold), i) for i, p in enumerate(unused) if p["name"] == gold["name"]]
        if scored:
            best, idx = max(scored)
            total += best
            unused.pop(idx)
    n_gold = len(gold_calls)
    extra = max(parsed.n_blocks - n_gold, 0)
    return total / (n_gold + extra)


def completion_text(completion) -> str:
    """TRL passes a string for plain prompts and a list of messages for chat prompts."""
    if isinstance(completion, str):
        return completion
    return "".join(m.get("content", "") for m in completion)


def call_reward(completions, gold_calls, **kwargs) -> list[float]:
    """TRL reward function. gold_calls is a column of JSON strings (Arrow cannot hold ragged dicts)."""
    return [reward_value(completion_text(c), json.loads(g)) for c, g in zip(completions, gold_calls)]


def format_reward(completions, **kwargs) -> list[float]:
    """1.0 when there is at least one block and every block is a well formed call."""
    out = []
    for c in completions:
        parsed = parse_calls(completion_text(c))
        out.append(1.0 if parsed.n_blocks > 0 and len(parsed.calls) == parsed.n_blocks else 0.0)
    return out
