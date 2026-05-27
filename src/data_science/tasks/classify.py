"""Classification pipeline: cross-validate every configured classifier with preprocessing."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_science.datasets import DATASETS
from data_science.viz import print_statistics as stats

_HYPERPARAMS_FILE = Path(__file__).resolve().parents[3] / "configs" / "hyperparameters.yaml"
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


def run(data, source):
    target = DATASETS[source].target_column
    data = data.apply(pd.to_numeric)
    y = data.pop(target).values.astype(int)
    X = data.values.astype(float)
    labels = np.unique(y)
    is_binary = len(labels) == 2

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    reports = []
    for cfg in CONFIGS:
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("smote", SMOTE(sampling_strategy="all", random_state=42)),
                ("clf", cfg.estimator_cls(**cfg.defaults(source))),
            ]
        )
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        reports.append(
            (
                cfg,
                accuracy_score(y, y_pred),
                recall_score(y, y_pred) if is_binary else None,
                confusion_matrix(y, y_pred, labels=labels),
            )
        )

    if is_binary:
        rows = [[cfg.name, cfg.defaults(source), acc, rec, cm] for cfg, acc, rec, cm in reports]
        stats.print_report(rows, (True, True))
    else:
        rows = [[cfg.name, [cfg.defaults(source), acc, cm]] for cfg, acc, _, cm in reports]
        stats.print_analysis_CT(rows, (True, True))
