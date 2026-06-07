"""End-to-end reproduce pipeline: stage_1 -> stage_2 -> sweeps -> final, for both datasets.

This module wires the four per-step CLI wrappers (each in its own module:
`stage_1.py`, `stage_2.py`, `sweeps.py`, `final.py`) into a single
`reproduce_all()` entrypoint that runs the full pipeline for every
config in `configs/{parkinsons,covertype}.yaml`.

The order is `stage_1 -> stage_2 -> sweeps -> final`:
- `stage_2` requires `stage_1_results.csv` written by `stage_1`.
- `sweeps` only needs `stage_1_results.csv` (its classifier-hyperparameter
  sub-step derives each classifier's best preprocessing from Stage 1, not
  from Stage 2 tuning), so it can run any time after `stage_1`.
- `final` requires `tuning_results.csv` written by `stage_2`.
"""

from __future__ import annotations

import time
from pathlib import Path

from cml_bench.experiments.final import run_final_evaluation
from cml_bench.experiments.stage_1 import run_stage_1
from cml_bench.experiments.stage_2 import run_stage_2
from cml_bench.experiments.sweeps import run_sweeps

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIGS = [
    REPO_ROOT / "configs" / "parkinsons.yaml",
    REPO_ROOT / "configs" / "covertype.yaml",
]


def reproduce_all(configs: list[Path] | None = None) -> None:
    """Run the full pipeline for each config in order: stage_1 -> stage_2 -> sweeps -> final."""
    if configs is None:
        configs = DEFAULT_CONFIGS

    overall_start = time.time()
    for config_path in configs:
        print()
        print("#" * 78)
        print(f"# Reproducing pipeline for: {config_path}")
        print("#" * 78)
        print()
        run_stage_1(config_path)
        print()
        run_stage_2(config_path)
        print()
        run_sweeps(config_path)
        print()
        run_final_evaluation(config_path)

    print()
    print(f"=== reproduce_all done in {(time.time() - overall_start) / 60:.1f} min ===")
