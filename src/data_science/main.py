import argparse

import yaml

from data_science.datasets import DATASETS
from data_science.tasks import arm, classify, clustering, preprocess
from data_science.tasks.experiment import run_experiment

TASKS = ("classify", "preprocess", "arm", "clustering", "experiment")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a task on a dataset.",
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        help="Required for all tasks except experiment.",
    )
    parser.add_argument(
        "--config",
        help="YAML config file (required for --task experiment).",
    )
    args = parser.parse_args()

    if args.task == "experiment":
        if not args.config:
            parser.error("--task experiment requires --config <path>")
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        run_experiment(**cfg)
        return

    if not args.dataset:
        parser.error(f"--task {args.task} requires --dataset")

    data = DATASETS[args.dataset].read()
    if args.task == "classify":
        classify.run(data, args.dataset)
    elif args.task == "preprocess":
        preprocess.run(data, args.dataset)
    elif args.task == "arm":
        arm.run(data, args.dataset)
    elif args.task == "clustering":
        clustering.run(data, args.dataset)


if __name__ == "__main__":
    main()
