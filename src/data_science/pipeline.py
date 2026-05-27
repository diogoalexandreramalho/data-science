from dataclasses import dataclass, field
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

_SWEEPS_FILE = Path(__file__).resolve().parents[2] / "configs" / "sweeps.yaml"
with _SWEEPS_FILE.open() as _f:
    SWEEPS = yaml.safe_load(_f)


@dataclass(frozen=True)
class ClassifierConfig:
    name: str
    estimator_cls: type
    call_params: dict = field(default_factory=dict)
    param_grid_key: str = ""  # keys into SWEEPS[source]
    display_params: tuple | str | None = None

    @property
    def display(self):
        if self.display_params is not None:
            return self.display_params
        return tuple(self.call_params.values())

    def param_grid(self, source: str) -> dict:
        return SWEEPS[source].get(self.param_grid_key, {})


# fmt: off
# CT display_params drift from call_params on DT/RF/GB/XGB — preserved as-is from the
# original; the printed report shows tuned values that aren't what training actually used.
PD_CONFIGS = [
    ClassifierConfig("Naive Bayes",       GaussianNB,                  {},                                                                                   "naive_bayes",     "GaussianNB"),
    ClassifierConfig("kNN",               KNeighborsClassifier,        {"n_neighbors": 1, "metric": "manhattan"},                                            "knn",             ("manhattan", 1)),
    ClassifierConfig("Decision Tree",     DecisionTreeClassifier,      {"min_samples_leaf": 0.05, "max_depth": 5, "criterion": "entropy"},                   "decision_tree",   ("entropy", 5, 0.05)),
    ClassifierConfig("Random Forest",     RandomForestClassifier,      {"n_estimators": 150, "max_depth": 10, "max_features": "sqrt"},                       "random_forest",   ("sqrt", 10, 150)),
    ClassifierConfig("Gradient Boosting", GradientBoostingClassifier,  {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "max_features": "sqrt"},  "gradient_boost",  ("sqrt", 5, 100, 0.1)),
    ClassifierConfig("XGBoost",           XGBClassifier,               {"n_estimators": 200, "max_depth": 5},                                                "xgboost",         (5, 200)),
]

CT_CONFIGS = [
    ClassifierConfig("Naive Bayes",       GaussianNB,                  {},                                                                                   "naive_bayes",     "GaussianNB"),
    ClassifierConfig("kNN",               KNeighborsClassifier,        {"n_neighbors": 1, "metric": "manhattan"},                                            "knn",             ("manhattan", 1)),
    ClassifierConfig("Decision Tree",     DecisionTreeClassifier,      {"min_samples_leaf": 0.05, "max_depth": 5, "criterion": "entropy"},                   "decision_tree",   ("entropy", 50, 0.00005)),
    ClassifierConfig("Random Forest",     RandomForestClassifier,      {"n_estimators": 150, "max_depth": 10, "max_features": "sqrt"},                       "random_forest",   ("sqrt", 25, 185)),
    ClassifierConfig("Gradient Boosting", GradientBoostingClassifier,  {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "max_features": "sqrt"},  "gradient_boost",  ("sqrt", 10, 300, 0.05)),
    ClassifierConfig("XGBoost",           XGBClassifier,               {"n_estimators": 200, "max_depth": 5},                                                "xgboost",         (10, 300)),
]
# fmt: on

CONFIGS_BY_SOURCE = {"PD": PD_CONFIGS, "CT": CT_CONFIGS}


def classification(data, source):
    target = DATASETS[source].target_column
    configs = CONFIGS_BY_SOURCE[source]

    # split features / target
    data = data.apply(pd.to_numeric)
    y: np.ndarray = data.pop(target).values.astype(int)
    X: np.ndarray = data.values.astype(float)
    labels: np.ndarray = pd.unique(y)
    n_classes = len(labels)
    is_binary = n_classes == 2

    # per-classifier accumulators (parallel to configs)
    accuracies: list[list[float]] = [[] for _ in configs]
    recalls: list[list[float]] = [[] for _ in configs]
    cnf_mtxs = [np.zeros((n_classes, n_classes), dtype=int) for _ in configs]

    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        trnX, tstX, trnY, tstY = X[train_index], X[test_index], y[train_index], y[test_index]
        trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
        trnX, trnY = balance.run(trnX, trnY, "all", 42, False)

        for i, cfg in enumerate(configs):
            estimator = cfg.estimator_cls(**cfg.call_params)
            acc, recall, cnf = train(estimator, trnX, tstX, trnY, tstY, labels)
            accuracies[i].append(acc)
            if recall is not None:
                recalls[i].append(recall)
            cnf_mtxs[i] = np.add(cnf_mtxs[i], cnf)

    avg_accuracies = [sum(a) / len(a) for a in accuracies]
    avg_recalls = [sum(r) / len(r) for r in recalls] if is_binary else None

    if is_binary:
        reports = [
            [cfg.name, cfg.display, avg_accuracies[i], avg_recalls[i], cnf_mtxs[i]]
            for i, cfg in enumerate(configs)
        ]
        stats.print_report(reports, (True, True))
    else:
        reports = [
            [cfg.name, [cfg.display, avg_accuracies[i], cnf_mtxs[i]]]
            for i, cfg in enumerate(configs)
        ]
        stats.print_analysis_CT(reports, (True, True))
