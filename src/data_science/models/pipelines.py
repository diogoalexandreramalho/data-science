from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from data_science.features.preprocessing import (
    build_feature_selector,
    build_pca_transformer,
    build_preprocessor,
)


def build_pipeline(
    X: pd.DataFrame,
    model: Any,
    preprocessing_config: dict[str, Any],
    continuous_columns: list[str] | None = None,
) -> Pipeline:
    """Compose a full sklearn Pipeline from a preprocessing config and model.

    Pure orchestration: the actual transformer construction lives in
    `features/preprocessing.py`. This function just reads the YAML config
    flags and assembles the steps in the canonical order:
    preprocessor -> (feature_selection) -> (pca) -> model.

    Keeping preprocessing inside the Pipeline ensures it is fitted only on
    training folds during cross-validation, preventing data leakage.
    """
    steps: list[tuple[str, Any]] = []

    steps.append(
        (
            "preprocessor",
            build_preprocessor(
                X=X,
                scale=preprocessing_config.get("scale", False),
                continuous_columns=continuous_columns,
            ),
        )
    )

    if preprocessing_config.get("feature_selection", False):
        k_best = preprocessing_config["k_best"]
        steps.append(("feature_selection", build_feature_selector(k_best=k_best)))

    if preprocessing_config.get("pca", False):
        pca_components = preprocessing_config["pca_components"]
        steps.append(("pca", build_pca_transformer(n_components=pca_components)))

    steps.append(("model", model))

    return Pipeline(steps)
