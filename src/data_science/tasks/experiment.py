"""Configurable experiment runner: pick dataset, scaler, sampler, classifier, action.

Use via the CLI (`--task experiment --config FILE`) or directly:
    from data_science.experiment import run_experiment
    run_experiment(dataset="PD", classifier="xgboost", action="sweep")
"""

import time
from datetime import timedelta

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_science.datasets import DATASETS
from data_science.tasks.classify import CONFIGS

ACTIONS = ("train", "sweep")

_CONFIGS_BY_KEY = {c.key: c for c in CONFIGS}
_CLASSIFIER_KEYS = tuple(_CONFIGS_BY_KEY)

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "none": None,
}

_SAMPLERS = {
    "smote": lambda: SMOTE(sampling_strategy="all", random_state=42),
    "none": None,
}


def _build_pipeline(dataset: str, classifier: str, scaler: str, sampler: str):
    if scaler not in _SCALERS:
        raise ValueError(f"unknown scaler: {scaler!r} (must be one of {list(_SCALERS)})")
    if sampler not in _SAMPLERS:
        raise ValueError(f"unknown sampler: {sampler!r} (must be one of {list(_SAMPLERS)})")
    if classifier not in _CLASSIFIER_KEYS:
        raise ValueError(
            f"unknown classifier: {classifier!r} (must be one of {list(_CLASSIFIER_KEYS)})"
        )

    steps = []
    if _SCALERS[scaler] is not None:
        steps.append(("scaler", _SCALERS[scaler]()))
    if _SAMPLERS[sampler] is not None:
        steps.append(("sampler", _SAMPLERS[sampler]()))

    cfg = _CONFIGS_BY_KEY[classifier]
    steps.append(("clf", cfg.estimator_cls(**cfg.defaults(dataset))))

    pipeline_cls = ImbPipeline if _SAMPLERS[sampler] is not None else Pipeline
    return pipeline_cls(steps)


def run_experiment(
    dataset: str,
    classifier: str,
    scaler: str = "standard",
    sampler: str = "smote",
    action: str = "train",
) -> None:
    """Run a single configurable experiment.

    Args:
        dataset:    "PD" or "CT"
        classifier: one of {naive_bayes, knn, decision_tree, random_forest, gradient_boost, xgboost}
        scaler:     "standard" | "minmax" | "none"
        sampler:    "smote" | "none"
        action:     "train" (single 10-fold CV) or "sweep" (GridSearchCV over hyperparams)
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action!r} (must be one of {ACTIONS})")
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset!r} (must be one of {list(DATASETS)})")

    print(
        f"=== Experiment: dataset={dataset}, classifier={classifier}, "
        f"scaler={scaler}, sampler={sampler}, action={action} ==="
    )

    data = DATASETS[dataset].read()
    target = DATASETS[dataset].target_column
    data = data.apply(pd.to_numeric)
    y = data.pop(target).values.astype(int)
    X = data.values.astype(float)
    labels = np.unique(y)
    is_binary = len(labels) == 2

    pipe = _build_pipeline(dataset, classifier, scaler, sampler)
    start = time.time()

    if action == "train":
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        elapsed = timedelta(seconds=time.time() - start)

        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        if is_binary:
            print(f"Recall:   {recall_score(y, y_pred):.4f}")
        print(f"Confusion matrix:\n{confusion_matrix(y, y_pred, labels=labels)}")
        print(f"Elapsed:  {elapsed}")

    elif action == "sweep":
        cfg = _CONFIGS_BY_KEY[classifier]
        grid = cfg.grid(dataset)
        if not grid:
            print(f"No hyperparameter grid defined for {classifier} — sweep is a no-op.")
            return
        # Prefix grid keys with "clf__" since classifier is the "clf" step in the Pipeline
        prefixed_grid = {f"clf__{k}": v for k, v in grid.items()}
        gs = GridSearchCV(pipe, prefixed_grid, scoring="accuracy", cv=10, n_jobs=-1)
        gs.fit(X, y)
        elapsed = timedelta(seconds=time.time() - start)

        # Strip "clf__" prefix from best_params for cleaner output
        best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        print(f"Best params: {best_params}")
        print(f"Best score:  {gs.best_score_:.4f}")
        print(f"Elapsed:     {elapsed}")
