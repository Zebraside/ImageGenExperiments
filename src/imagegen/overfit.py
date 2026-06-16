"""Entry point for the single-batch overfit sanity check.

Equivalent to:
    uv run python -m imagegen.train --config configs/overfit.yaml [overrides...]

Any extra dotlist overrides are forwarded to the trainer, e.g.:
    uv run imagegen-overfit train.max_steps=50
"""

import sys

from imagegen.train import main as _train_main


def main() -> None:
    sys.argv = [sys.argv[0], "--config", "configs/overfit.yaml"] + sys.argv[1:]
    _train_main()
