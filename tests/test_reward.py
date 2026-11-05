import json

import pytest

from toolcall_grpo.calls import parse_calls
from toolcall_grpo.evaluate import score_outputs
from toolcall_grpo.reward import call_reward, format_reward


def block(name, **arguments):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


WEATHER = [{"name": "get_weather", "arguments": {"city": "KL", "unit": "c"}}]
TWO = [
    {"name": "a", "arguments": {"x": 1}},
    {"name": "b", "arguments": {"y": "z"}},
]


@pytest.mark.parametrize(
    "completion, gold, expected",
    [
        (block("get_weather", city="KL", unit="c"), WEATHER, 1.0),
        ("It is sunny.", WEATHER, 0.0),  # no call at all
        ("<tool_call>\n{not json}\n</tool_call>", WEATHER, 0.0),  # block that does not parse
        (block("get_forecast", city="KL", unit="c"), WEATHER, 0.0),  # wrong function
        (block("get_weather"), WEATHER, 0.3),  # right function, no arguments: name credit only
        (block("get_weather", city="KL"), WEATHER, 0.3 + 0.7 * (2 / 3)),  # one of two arguments
        (block("get_weather", city="KL", unit="c", extra=1), WEATHER, 0.3 + 0.7 * 0.8),  # extra argument
        (block("get_weather", city=" KL ", unit="c"), WEATHER, 1.0),  # whitespace ignored
        (block("a", x=1.0) + block("b", y="z"), TWO, 1.0),  # 1 and 1.0 match
        (block("b", y="z") + block("a", x=1), TWO, 1.0),  # parallel calls in any order
        (block("a", x=1), TWO, 0.5),  # one of two calls missing
        (block("get_weather", city="KL", unit="c") * 2, WEATHER, 0.5),  # a repeated call is penalised
        (block("f", flag=1), [{"name": "f", "arguments": {"flag": True}}], 0.3),  # True is not 1
    ],
)
def test_call_reward(completion, gold, expected):
    assert call_reward([completion], [json.dumps(gold)]) == [pytest.approx(expected)]


def test_format_reward_requires_every_block_to_be_a_well_formed_call():
    good, bad = block("f", a=1), "<tool_call>\n{broken\n</tool_call>"
    assert format_reward([good, good + bad, "no call", bad]) == [1.0, 0.0, 0.0, 0.0]


def test_gold_labels_in_python_style_parse_only_in_lenient_mode():
    # The dataset writes some labels as Python dicts, wrapped in a literal backslash n.
    text = "<tool_call>\\n{'name': 'f', 'arguments': {'q': ['a', 'b']}}\\n</tool_call>"
    assert parse_calls(text).calls == [] and parse_calls(text).n_blocks == 1
    assert parse_calls(text, lenient=True).calls == [{"name": "f", "arguments": {"q": ["a", "b"]}}]


def test_score_outputs_reports_each_metric_and_the_split_by_call_count():
    one = [{"name": "a", "arguments": {"x": 1}}]
    two = [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]
    examples = [{"id": str(i), "gold_calls": json.dumps(g)} for i, g in enumerate([one, one, two])]
    texts = [
        block("a", x=1),  # exact
        "no call",  # nothing
        block("a", x=2) + block("c"),  # right name once, wrong value, one stray call
    ]
    rows, summary = score_outputs(examples, texts)
    assert [r["exact"] for r in rows] == [True, False, False]
    assert summary["exact_match"] == pytest.approx(1 / 3)
    assert summary["well_formed"] == pytest.approx(2 / 3)
    assert summary["name_recall"] == pytest.approx((1 + 0 + 0.5) / 3)
    assert summary["by_number_of_gold_calls"]["1 call"]["n"] == 2
    assert summary["by_number_of_gold_calls"]["2 calls"]["mean_reward"] == pytest.approx(0.3 / 2)


def test_compare_counts_changes_and_gives_an_interval_that_excludes_zero_for_a_consistent_gain():
    from toolcall_grpo.compare import compare

    before = {str(i): {"reward": 0.5, "exact": False} for i in range(40)}
    after = {str(i): {"reward": 0.7, "exact": i < 10} for i in range(40)}
    after["0"]["reward"] = 0.4  # one example got worse
    result = compare(before, after)
    assert (result["better"], result["worse"], result["same"]) == (39, 1, 0)
    assert result["mean_change"] == pytest.approx((39 * 0.2 - 0.1) / 40)
    assert result["exact_after"] == pytest.approx(10 / 40)
    low, high = result["ci95"]
    assert 0 < low < result["mean_change"] < high


def test_hint_goes_inside_the_user_turn_and_a_prompt_without_that_ending_is_refused():
    from toolcall_grpo.evaluate import HINT, add_hint

    prompt = "<|im_start|>user\nBook it.<|im_end|>\n<|im_start|>assistant\n"
    assert add_hint(prompt) == f"<|im_start|>user\nBook it.\n\n{HINT}<|im_end|>\n<|im_start|>assistant\n"
    with pytest.raises(ValueError):
        add_hint("<|im_start|>user\nBook it.<|im_end|>")
