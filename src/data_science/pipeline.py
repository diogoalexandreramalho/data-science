from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data_science.datasets import DATASETS
from data_science.models import decision_tree as dt
from data_science.models import gradient_boost as gb
from data_science.models import knn
from data_science.models import naive_bayes as nb
from data_science.models import random_forest as rf
from data_science.models import xgboost_clf as xgb
from data_science.preprocessing import data_balancing as balance
from data_science.preprocessing import normalize as norm
from data_science.viz import print_statistics as stats


@dataclass(frozen=True)
class ClassifierConfig:
    name: str
    fn: Callable
    call_params: tuple = ()
    display_params: tuple | None = None

    @property
    def display(self) -> tuple:
        return self.display_params if self.display_params is not None else self.call_params


# fmt: off
# CT display_params drift from call_params on DT/RF/GB/XGB — preserved as-is from the
# original; the printed report shows tuned values that aren't what training actually used.
PD_CONFIGS = [
    ClassifierConfig("Naive Bayes",       nb.simple_naive_bayes,    (),                       "GaussianNB"),
    ClassifierConfig("kNN",               knn.simple_knn,           (1, "manhattan"),         ("manhattan", 1)),
    ClassifierConfig("Decision Tree",     dt.simple_decision_tree,  (0.05, 5, "entropy"),     ("entropy", 5, 0.05)),
    ClassifierConfig("Random Forest",     rf.simple_random_forest,  (150, 10, "sqrt"),        ("sqrt", 10, 150)),
    ClassifierConfig("Gradient Boosting", gb.simple_gradient_boost, (100, 0.1, 5, "sqrt"),    ("sqrt", 5, 100, 0.1)),
    ClassifierConfig("XGBoost",           xgb.simple_xg_boost,      (200, 5),                 (5, 200)),
]

CT_CONFIGS = [
    ClassifierConfig("Naive Bayes",       nb.simple_naive_bayes_CT,    (),                    "GaussianNB"),
    ClassifierConfig("kNN",               knn.simple_knn_CT,           (1, "manhattan"),      ("manhattan", 1)),
    ClassifierConfig("Decision Tree",     dt.simple_decision_tree_CT,  (0.05, 5, "entropy"),  ("entropy", 50, 0.00005)),
    ClassifierConfig("Random Forest",     rf.simple_random_forest_CT,  (150, 10, "sqrt"),     ("sqrt", 25, 185)),
    ClassifierConfig("Gradient Boosting", gb.simple_gradient_boost_CT, (100, 0.1, 5, "sqrt"), ("sqrt", 10, 300, 0.05)),
    ClassifierConfig("XGBoost",           xgb.simple_xg_boost_CT,      (200, 5),              (10, 300)),
]
# fmt: on

CONFIGS_BY_SOURCE = {"PD": PD_CONFIGS, "CT": CT_CONFIGS}


def classification(data, source):
    target = DATASETS[source].target_column
    configs = CONFIGS_BY_SOURCE[source]
    is_binary = source == "PD"

    # split features / target
    data = data.apply(pd.to_numeric)
    y: np.ndarray = data.pop(target).values.astype(int)
    X: np.ndarray = data.values.astype(float)
    labels: np.ndarray = pd.unique(y)
    n_classes = len(labels)

    # per-classifier accumulators (parallel to configs)
    accuracies: list[list[float]] = [[] for _ in configs]
    specificities: list[list[float]] = [[] for _ in configs]
    cnf_mtxs = [np.zeros((n_classes, n_classes), dtype=int) for _ in configs]

    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        trnX, tstX, trnY, tstY = X[train_index], X[test_index], y[train_index], y[test_index]
        trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
        trnX, trnY = balance.run(trnX, trnY, "all", 42, False)

        for i, cfg in enumerate(configs):
            if is_binary:
                acc, spec, cnf = cfg.fn(trnX, tstX, trnY, tstY, *cfg.call_params, labels)
                specificities[i].append(spec)
            else:
                acc, cnf = cfg.fn(trnX, tstX, trnY, tstY, *cfg.call_params, labels)
            accuracies[i].append(acc)
            cnf_mtxs[i] = np.add(cnf_mtxs[i], cnf)

    avg_accuracies = [sum(a) / len(a) for a in accuracies]
    avg_specificities = [sum(s) / len(s) for s in specificities] if is_binary else None

    if is_binary:
        reports = [
            [cfg.name, cfg.display, avg_accuracies[i], avg_specificities[i], cnf_mtxs[i]]
            for i, cfg in enumerate(configs)
        ]
        stats.print_report(reports, (True, True))
    else:
        reports = [
            [cfg.name, [cfg.display, avg_accuracies[i], cnf_mtxs[i]]]
            for i, cfg in enumerate(configs)
        ]
        stats.print_analysis_CT(reports, (True, True))
