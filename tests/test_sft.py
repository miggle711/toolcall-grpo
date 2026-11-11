import json
from pathlib import Path

import pytest
import yaml

from toolcall_grpo.reward import reward_value
from toolcall_grpo.train_sft import render_completion

pytest.importorskip("trl")

from trl import SFTConfig  # noqa: E402


def test_rendered_gold_completion_scores_full_reward_under_the_grader():
    gold = [{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_time", "arguments": {"tz": "UTC"}}]
    assert reward_value(render_completion(json.dumps(gold)), gold) == 1.0


def test_sft_config_builds_with_loss_on_the_completion_only(tmp_path):
    cfg = yaml.safe_load((Path(__file__).parent.parent / "configs/sft/sft_1p5b.yaml").read_text())
    cfg["sft"]["bf16"] = False  # the machine running the tests may have no GPU
    assert SFTConfig(output_dir=str(tmp_path), **cfg["sft"]).completion_only_loss is True
