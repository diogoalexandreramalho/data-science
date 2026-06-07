"""Shared bootstrap for config-driven experiment subcommands.

Every CLI subcommand (`stage-1`, `sweeps`, `tune`, `final`) starts from the
same setup: load YAML config, read raw dataset, optionally subsample, split
train/test, build the CV strategy, resolve scoring metrics, instantiate
models. This module centralises that bootstrap into a single
`load_context()` call returning a `Context` dataclass.

Module-private (leading underscore in the filename): only the per-step
wrapper modules in `experiments/` and `experiments/reproduce.py` import
from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data_science.data.datasets import DATASETS
from data_science.data.sampling import create_balanced_sample
from data_science.data.split import create_cv_strategy, create_train_test_split
from data_science.evaluation.metrics import resolve_scoring
from data_science.models.registry import get_model_registry
from data_science.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
HYPERPARAMS_PATH = REPO_ROOT / "configs" / "hyperparameters.yaml"

# hyperparameters.yaml uses "gradient_boost" while the model registry uses
# "gradient_boosting"; remap when looking up grids.
HYPERPARAMS_CLASSIFIER_REMAP = {"gradient_boosting": "gradient_boost"}


@dataclass
class Context:
    cfg: dict[str, Any]
    name: str
    source_code: str
    output_dir: Path

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series | None
    groups_test: pd.Series | None

    cv: Any
    scoring: dict[str, Any]
    primary_metric: str
    models: dict[str, Any]
    continuous_columns: list[str] | None

    class_names: list[str]
    is_binary: bool


def load_context(config_path: str | Path) -> Context:
    """Read config, load + split data, build CV, return a bootstrap context."""
    cfg = load_config(config_path)

    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"]
    source_code = dataset_cfg["source_code"]
    target_column = dataset_cfg["target_column"]
    group_column = dataset_cfg.get("group_column")
    drop_columns = dataset_cfg.get("drop_columns", [])
    class_names = dataset_cfg["class_names"]
    zero_index_labels = dataset_cfg.get("zero_index_labels", False)
    is_binary = len(class_names) == 2

    random_state = cfg["split"]["random_state"]

    raw = DATASETS[source_code].read()
    sampling = cfg.get("sampling", {})
    if sampling.get("balanced_subsample", False):
        df = create_balanced_sample(
            df=raw,
            target_column=target_column,
            samples_per_class=sampling["samples_per_class"],
            random_state=random_state,
        )
    else:
        df = raw

    y_raw = df[target_column]
    y = y_raw - 1 if zero_index_labels else y_raw
    groups = df[group_column] if group_column else None
    X = df.drop(columns=[target_column] + drop_columns)

    if cfg["split"]["strategy"] == "group":
        X_train, X_test, y_train, y_test, g_train, g_test = create_train_test_split(
            X,
            y,
            strategy="group",
            test_size=cfg["split"]["test_size"],
            random_state=random_state,
            groups=groups,
        )
    else:
        X_train, X_test, y_train, y_test, _, _ = create_train_test_split(
            X,
            y,
            strategy=cfg["split"]["strategy"],
            test_size=cfg["split"]["test_size"],
            random_state=random_state,
        )
        g_train = None
        g_test = None

    cv = create_cv_strategy(
        strategy=cfg["cv"]["strategy"],
        n_splits=cfg["cv"]["n_splits"],
        random_state=random_state,
    )

    scoring = resolve_scoring(cfg["metrics"]["scoring"])
    primary_metric = cfg["metrics"]["primary"]
    continuous_columns = cfg.get("columns", {}).get("continuous")

    registry = get_model_registry(random_state)
    models = {model_name: registry[model_name] for model_name in cfg["models"]}

    output_dir = REPO_ROOT / "artifacts" / "final" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    return Context(
        cfg=cfg,
        name=name,
        source_code=source_code,
        output_dir=output_dir,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        groups_train=g_train,
        groups_test=g_test,
        cv=cv,
        scoring=scoring,
        primary_metric=primary_metric,
        models=models,
        continuous_columns=continuous_columns,
        class_names=class_names,
        is_binary=is_binary,
    )


def load_grids(source_code: str, classifier_names: list[str]) -> dict[str, dict[str, list[Any]]]:
    """Load Stage-2 grid-search hyperparameter grids from configs/hyperparameters.yaml."""
    with HYPERPARAMS_PATH.open() as f:
        all_grids = yaml.safe_load(f)

    grids: dict[str, dict[str, list[Any]]] = {}
    for model_name in classifier_names:
        key = HYPERPARAMS_CLASSIFIER_REMAP.get(model_name, model_name)
        if key in all_grids[source_code]:
            grids[model_name] = all_grids[source_code][key].get("grid", {})
    return grids
