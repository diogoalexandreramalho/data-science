import argparse

from data_science.datasets import DATASETS
from data_science.tasks import arm, classify, clustering, preprocess

TASKS = ("classify", "preprocess", "arm", "clustering")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a task (classify / preprocess / arm / clustering) on a dataset.",
    )
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()

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
