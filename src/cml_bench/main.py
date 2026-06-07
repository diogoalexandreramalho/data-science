from __future__ import annotations

import argparse

from cml_bench.data.datasets import DATASETS
from cml_bench.experiments.final import run_final_evaluation
from cml_bench.experiments.reproduce import reproduce_all
from cml_bench.experiments.stage_1 import run_stage_1
from cml_bench.experiments.stage_2 import run_stage_2
from cml_bench.experiments.sweeps import run_sweeps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the ML report.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--dataset", choices=list(DATASETS), required=True)

    stage_1 = subparsers.add_parser("stage-1")
    stage_1.add_argument("--config", required=True)

    sweeps = subparsers.add_parser("sweeps")
    sweeps.add_argument("--config", required=True)

    stage_2 = subparsers.add_parser("stage-2")
    stage_2.add_argument("--config", required=True)

    final = subparsers.add_parser("final")
    final.add_argument("--config", required=True)

    subparsers.add_parser("reproduce")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        dataset = DATASETS[args.dataset]
        dataset.download()
        print(f"Downloaded {dataset.name} to {dataset.raw_path}")
        return

    if args.command == "stage-1":
        run_stage_1(args.config)
        return

    if args.command == "sweeps":
        run_sweeps(args.config)
        return

    if args.command == "stage-2":
        run_stage_2(args.config)
        return

    if args.command == "final":
        run_final_evaluation(args.config)
        return

    if args.command == "reproduce":
        reproduce_all()
        return


if __name__ == "__main__":
    main()
