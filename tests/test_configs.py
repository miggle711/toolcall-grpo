from pathlib import Path

import pytest
import yaml

pytest.importorskip("trl")

from toolcall_grpo.train import build_configs  # noqa: E402

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_builds_the_real_trainer_settings_with_a_valid_generation_batch(path, tmp_path):
    cfg = yaml.safe_load(path.read_text())
    cfg["grpo"]["bf16"] = False  # the machine running the tests may have no GPU
    grpo, lora = build_configs(cfg, tmp_path)  # raises on any setting this TRL version does not accept
    # One generation batch must hold whole groups, or TRL refuses to start.
    assert grpo.generation_batch_size % grpo.num_generations == 0
    assert len(grpo.reward_weights) == 2  # format_reward, call_reward
