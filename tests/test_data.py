import json

import pytest

from toolcall_grpo.data import convert_row, split_by_tool_group


def row(call_text, tool_names=("f",), rid="r"):
    tools = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in tool_names]
    return {
        "id": rid,
        "category": "c",
        "tools": json.dumps(tools),
        "conversations": [
            {"from": "system", "value": "ignored"},
            {"from": "human", "value": "do it"},
            {"from": "gpt", "value": call_text},
        ],
    }


def render(messages, tools):
    return f"{messages[0]['content']}|{len(tools)} tools"


@pytest.mark.parametrize(
    "call_text, tool_names, usable",
    [
        ('<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>', ("f",), True),
        ("<tool_call>\\n{'name': 'f', 'arguments': {}}\\n</tool_call>", ("f",), True),  # Python style label
        ("just text", ("f",), False),  # no call
        # Part of the dataset puts the function name inside the arguments. Not repaired, dropped.
        ("<tool_call>\\n{'arguments': {'q': [], 'name': 'f'}}\\n</tool_call>", ("f",), False),
        ('<tool_call>\n{"name": "g", "arguments": {}}\n</tool_call>', ("f",), False),  # tool not offered
        ('<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call><tool_call>\nbroken\n</tool_call>', ("f",), False),
    ],
)
def test_convert_row_keeps_only_rows_with_fully_usable_gold_calls(call_text, tool_names, usable):
    ex = convert_row(row(call_text, tool_names), render)
    assert (ex is not None) is usable
    if usable:
        assert ex["prompt"] == "do it|1 tools"
        assert json.loads(ex["gold_calls"]) == [{"name": "f", "arguments": {}}]


def test_split_never_puts_a_tool_name_on_both_sides():
    # a and b are linked through one example that uses both, so they must stay together.
    names = [["a"], ["b"], ["a", "b"], ["c"], ["d"], ["e", "f"], ["f"], ["g"]] * 6
    examples = [{"id": i, "tool_names": n} for i, n in enumerate(names)]
    train, test = split_by_tool_group(examples, test_fraction=0.4, seed=1)
    assert len(train) + len(test) == len(examples)
    assert test and train
    assert {n for e in train for n in e["tool_names"]}.isdisjoint({n for e in test for n in e["tool_names"]})
    assert split_by_tool_group(examples, 0.4, 1) == (train, test)  # deterministic


def test_training_prompts_carry_the_hint_only_when_asked(tmp_path):
    from toolcall_grpo.evaluate import HINT
    from toolcall_grpo.train import load_examples

    path = tmp_path / "train.jsonl"
    prompt = "<|im_start|>user\nBook it.<|im_end|>\n<|im_start|>assistant\n"
    path.write_text(json.dumps({"prompt": prompt, "gold_calls": "[]"}) + "\n")
    assert load_examples(path, None)[0]["prompt"] == prompt
    assert HINT in load_examples(path, None, hint=True)[0]["prompt"]
