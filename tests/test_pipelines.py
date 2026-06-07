"""Smoke test: every preprocessing config from both YAMLs composes into a
runnable sklearn Pipeline that fits + predicts without error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_science.models.pipelines import build_pipeline
from data_science.models.registry import get_model_registry
from data_science.utils.config import load_config


def _synthetic_xy(n_samples: int = 60, n_features: int = 12, n_classes: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.integers(0, n_classes, size=n_samples), name="label")
    return X, y


@pytest.mark.parametrize("config_name", ["parkinsons.yaml", "covertype.yaml"])
def test_every_preprocessing_config_fits_and_predicts(repo_root, config_name):
    cfg = load_config(repo_root / "configs" / config_name)
    n_classes = len(cfg["dataset"]["class_names"])
    X, y = _synthetic_xy(n_classes=n_classes)

    registry = get_model_registry(random_state=42)
    # Use a single fast classifier for the smoke test.
    model = registry["decision_tree"]

    for preproc_cfg in cfg["preprocessing"]["configs"]:
        # Force k_best and pca_components to fit the tiny synthetic data
        cfg_for_smoke = dict(preproc_cfg)
        if cfg_for_smoke.get("feature_selection"):
            cfg_for_smoke["k_best"] = min(cfg_for_smoke["k_best"], X.shape[1] - 1)
        if cfg_for_smoke.get("pca"):
            cfg_for_smoke["pca_components"] = 0.95

        pipe = build_pipeline(X=X, model=model, preprocessing_config=cfg_for_smoke)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset(set(range(n_classes)))


def test_model_registry_exposes_all_required_classifiers():
    registry = get_model_registry(random_state=42)
    for name in (
        "naive_bayes",
        "knn",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    ):
        assert name in registry, f"missing classifier: {name}"
