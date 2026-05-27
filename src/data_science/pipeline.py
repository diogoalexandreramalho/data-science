from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_science.datasets import DATASETS
from data_science.preprocessing import data_balancing as balance
from data_science.preprocessing import normalize as norm
from data_science.train import train
from data_science.viz import print_statistics as stats

_HYPERPARAMS_FILE = Path(__file__).resolve().parents[2] / "configs" / "hyperparameters.yaml"
with _HYPERPARAMS_FILE.open() as _f:
    HYPERPARAMS = yaml.safe_load(_f)


@dataclass(frozen=True)
class ClassifierConfig:
    name: str
    estimator_cls: type
    key: str  # looks up HYPERPARAMS[source][key]

    def defaults(self, source: str) -> dict:
        return HYPERPARAMS[source][self.key]["defaults"]

    def grid(self, source: str) -> dict:
        return HYPERPARAMS[source][self.key]["grid"]


CONFIGS = [
    ClassifierConfig("Naive Bayes", GaussianNB, "naive_bayes"),
    ClassifierConfig("kNN", KNeighborsClassifier, "knn"),
    ClassifierConfig("Decision Tree", DecisionTreeClassifier, "decision_tree"),
    ClassifierConfig("Random Forest", RandomForestClassifier, "random_forest"),
    ClassifierConfig("Gradient Boosting", GradientBoostingClassifier, "gradient_boost"),
    ClassifierConfig("XGBoost", XGBClassifier, "xgboost"),
]

_CONFIGS_BY_KEY = {c.key: c for c in CONFIGS}


def get_classifier(key: str, source: str):
    """Instantiate a configured classifier (by YAML key) with its defaults for the given source."""
    cfg = _CONFIGS_BY_KEY[key]
    return cfg.estimator_cls(**cfg.defaults(source))


def classification(data, source):
    target = DATASETS[source].target_column

    # split features / target
    data = data.apply(pd.to_numeric)
    y: np.ndarray = data.pop(target).values.astype(int)
    X: np.ndarray = data.values.astype(float)
    labels: np.ndarray = pd.unique(y)
    n_classes = len(labels)
    is_binary = n_classes == 2

    # per-classifier accumulators (parallel to CONFIGS)
    accuracies: list[list[float]] = [[] for _ in CONFIGS]
    recalls: list[list[float]] = [[] for _ in CONFIGS]
    cnf_mtxs = [np.zeros((n_classes, n_classes), dtype=int) for _ in CONFIGS]

    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = (
            X[train_index],
            X[test_index],
            y[train_index],
            y[test_index],
        )
        X_train, X_test, y_train, y_test = norm.standard_scaler(X_train, X_test, y_train, y_test)
        X_train, y_train = balance.run(X_train, y_train, "all", 42, False)

        for i, cfg in enumerate(CONFIGS):
            estimator = cfg.estimator_cls(**cfg.defaults(source))
            acc, recall, cnf = train(estimator, X_train, X_test, y_train, y_test, labels)
            accuracies[i].append(acc)
            if recall is not None:
                recalls[i].append(recall)
            cnf_mtxs[i] = np.add(cnf_mtxs[i], cnf)

    avg_accuracies = [sum(a) / len(a) for a in accuracies]
    avg_recalls = [sum(r) / len(r) for r in recalls] if is_binary else None

    if is_binary:
        reports = [
            [cfg.name, cfg.defaults(source), avg_accuracies[i], avg_recalls[i], cnf_mtxs[i]]
            for i, cfg in enumerate(CONFIGS)
        ]
        stats.print_report(reports, (True, True))
    else:
        reports = [
            [cfg.name, [cfg.defaults(source), avg_accuracies[i], cnf_mtxs[i]]]
            for i, cfg in enumerate(CONFIGS)
        ]
        stats.print_analysis_CT(reports, (True, True))
