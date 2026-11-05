"""Download models and datasets into the project's HF cache.

Run on the login node (compute nodes have no internet), with HF_HOME pointing at scratch.
Jobs then run with HF_HUB_OFFLINE=1.
"""

import os
import sys

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # for smoke tests
    "Qwen/Qwen2.5-1.5B-Instruct",  # the model being trained
]
DATASETS = [
    "glaiveai/glaive-function-calling-v2",
    "NousResearch/hermes-function-calling-v1",
    "gorilla-llm/Berkeley-Function-Calling-Leaderboard",  # evaluation only
]


def main() -> None:
    if "HF_HOME" not in os.environ:
        sys.exit("Set HF_HOME to a scratch path first, home has too little space.")
    for repo in MODELS:
        print("model  ", repo, snapshot_download(repo_id=repo))
    for repo in DATASETS:
        print("dataset", repo, snapshot_download(repo_id=repo, repo_type="dataset"))


if __name__ == "__main__":
    main()
