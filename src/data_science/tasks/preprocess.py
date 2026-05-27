"""Compare preprocessing techniques across the two datasets.

Research/exploration code. Each *_analysis() function trains a few classifiers under
different preprocessing variants (balance, normalize, feature selection, PCA) using
sklearn Pipelines + cross_validate, and prints accuracy (and recall for PD).

Re-run by uncommenting one of the lines at the bottom and running:
    python scripts/preprocessing.py
"""

import math
import warnings

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, f_classif
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_science.tasks.classify import get_classifier

CLASSIFIERS = ["naive_bayes", "knn", "decision_tree"]


def _scoring(name):
    return (
        {"accuracy": "accuracy", "recall": "recall"} if name == "PD" else {"accuracy": "accuracy"}
    )


def _split_xy(dataset, name):
    """Return (X, y) for the given dataset variant; subsample CT for tractable runs."""
    data = dataset.copy()
    if name == "PD":
        data.pop("id")
        y = data.pop("class").values
    else:
        data = data.groupby("Cover_Type").apply(lambda s: s.sample(100))
        y = data.pop("Cover_Type").values
    return data.values, y


def _eval(pipe, X, y, name):
    """Run 10-fold CV; return (mean accuracy, mean recall or None)."""
    scores = cross_validate(pipe, X, y, cv=10, scoring=_scoring(name), n_jobs=-1)
    acc = scores["test_accuracy"].mean()
    rec = scores["test_recall"].mean() if name == "PD" else None
    return acc, rec


def _print_row(label, acc, rec):
    suffix = f", rec={rec:.3f}" if rec is not None else ""
    print(f"  {label} | acc={acc:.3f}{suffix}")


def balance_analysis(dataset, name):
    """Compare SMOTE-balanced vs unbalanced training."""
    X, y = _split_xy(dataset, name)
    print(f"\n=== Balance Analysis on {name} ===")
    for clf in CLASSIFIERS:
        for variant, sampler in [
            ("SMOTE", SMOTE(sampling_strategy="all", random_state=42)),
            ("None", None),
        ]:
            steps = [("scale", StandardScaler())]
            if sampler is not None:
                steps.append(("smote", sampler))
            steps.append(("clf", get_classifier(clf, name)))
            pipe = ImbPipeline(steps) if sampler is not None else Pipeline(steps)
            acc, rec = _eval(pipe, X, y, name)
            _print_row(f"{clf:14} | balance={variant:5}", acc, rec)


def normalize_analysis(dataset, name):
    """Compare MinMaxScaler vs StandardScaler vs no normalization."""
    X, y = _split_xy(dataset, name)
    print(f"\n=== Normalize Analysis on {name} ===")
    for clf in CLASSIFIERS:
        for variant, scaler in [
            ("min_max", MinMaxScaler()),
            ("standard", StandardScaler()),
            ("None", None),
        ]:
            steps = []
            if scaler is not None:
                steps.append(("scale", scaler))
            steps.append(("smote", SMOTE(sampling_strategy="all", random_state=42)))
            steps.append(("clf", get_classifier(clf, name)))
            pipe = ImbPipeline(steps)
            acc, rec = _eval(pipe, X, y, name)
            _print_row(f"{clf:14} | norm={variant:8}", acc, rec)


def feature_selection_analysis(dataset, name):
    """Compare SelectKBest, SelectPercentile, and ExtraTrees-based feature selection."""
    X, y = _split_xy(dataset, name)
    print(f"\n=== Feature Selection Analysis on {name} ===")
    if name == "CT":
        warnings.filterwarnings("ignore")

    n_features_kbest = (2 ** np.arange(math.floor(math.log(X.shape[1], 2)) + 1)).tolist()
    n_features_kbest.append(X.shape[1])
    percentiles = (np.arange(1, 11) * 10).tolist()

    def _pipeline(selector, clf):
        return ImbPipeline(
            [
                ("scale", StandardScaler()),
                ("select", selector),
                ("smote", SMOTE(sampling_strategy="all", random_state=42)),
                ("clf", get_classifier(clf, name)),
            ]
        )

    for clf in ["naive_bayes", "knn"]:
        print(f"  -- SelectKBest, {clf} --")
        for k in n_features_kbest:
            acc, rec = _eval(_pipeline(SelectKBest(f_classif, k=k), clf), X, y, name)
            _print_row(f"k={k:4}", acc, rec)

        print(f"  -- SelectPercentile, {clf} --")
        for p in percentiles:
            acc, rec = _eval(_pipeline(SelectPercentile(f_classif, percentile=p), clf), X, y, name)
            _print_row(f"p={p:3}", acc, rec)

        print(f"  -- Wrapper (SelectFromModel/ExtraTrees), {clf} --")
        wrapper = SelectFromModel(ExtraTreesClassifier(n_estimators=50, random_state=42))
        acc, rec = _eval(_pipeline(wrapper, clf), X, y, name)
        _print_row("wrapper", acc, rec)


def pca_analysis(dataset, name):
    """Compare PCA with varying number of components."""
    X, y = _split_xy(dataset, name)
    print(f"\n=== PCA Analysis on {name} ===")
    n_features = X.shape[1]
    components = [max(1, int(n_features * p)) for p in (0.1, 0.25, 0.5, 0.75)]

    for clf in CLASSIFIERS:
        for n in components:
            pipe = ImbPipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=n, random_state=42)),
                    ("smote", SMOTE(sampling_strategy="all", random_state=42)),
                    ("clf", get_classifier(clf, name)),
                ]
            )
            acc, rec = _eval(pipe, X, y, name)
            _print_row(f"{clf:14} | n_components={n:4}", acc, rec)


def run(data, source):
    """Run the full preprocessing comparison study on a dataset."""
    balance_analysis(data, source)
    normalize_analysis(data, source)
    feature_selection_analysis(data, source)
    if source == "PD":
        pca_analysis(data, source)
    else:
        print("\n(PCA skipped for CT — runtime impractical at full feature count)")
