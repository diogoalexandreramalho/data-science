"""Run an XGBoost hyperparameter sweep on a dataset and print the best config."""

import datetime
import time

from sklearn.model_selection import train_test_split

from data_science.datasets import DATASETS
from data_science.preprocessing import data_balancing as balance
from data_science.preprocessing import normalize as norm
from data_science.tasks.classify import CONFIGS
from data_science.train import sweep

_CONFIGS_BY_KEY = {c.key: c for c in CONFIGS}


def run_xgb_sweep(source: str) -> None:
    """Preprocess the dataset (split + normalize + SMOTE-balance), then sweep XGBoost."""
    data = DATASETS[source].read()
    target = DATASETS[source].target_column

    if source == "CT":
        # subsample 1000 per class for a tractable sweep
        data = data.groupby(target).apply(lambda s: s.sample(1000))

    y = data.pop(target).values
    X = data.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)
    X_train, X_test, y_train, y_test = norm.standard_scaler(X_train, X_test, y_train, y_test)
    X_train, y_train = balance.run(X_train, y_train, "all", 42, False)

    xgb_cfg = _CONFIGS_BY_KEY["xgboost"]

    start = time.time()
    result = sweep(xgb_cfg.estimator_cls, X_train, y_train, xgb_cfg.grid(source))
    elapsed = str(datetime.timedelta(seconds=time.time() - start))

    print(f"=== XGBoost sweep on {source} ===")
    print(f"Best params:  {result.best_params_}")
    print(f"Best score:   {result.best_score_:.4f}")
    print(f"Elapsed:      {elapsed}")


if __name__ == "__main__":
    run_xgb_sweep("CT")
