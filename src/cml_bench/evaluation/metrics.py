from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Custom scorers for metrics sklearn does not expose as a named string scorer.
# Specificity is recall on the negative class (true-negative rate) — i.e. how
# many actual healthy controls were correctly classified as healthy.
CUSTOM_SCORERS = {
    "specificity": make_scorer(recall_score, pos_label=0, zero_division=0),
}


def resolve_scoring(scoring: dict[str, str]) -> dict[str, Any]:
    """Translate a YAML scoring dict (string-valued) into sklearn-ready scorers.

    Sklearn-named strings (e.g. "f1", "roc_auc") pass through unchanged; custom
    scorer names (e.g. "specificity") are replaced with their scorer objects.
    """
    resolved: dict[str, Any] = {}
    for friendly_name, scorer_name in scoring.items():
        resolved[friendly_name] = CUSTOM_SCORERS.get(scorer_name, scorer_name)
    return resolved


def evaluate_binary_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute the binary-classification metric set against a held-out test set.

    Returns accuracy, precision, recall (sensitivity on the positive class),
    specificity (recall on the negative class), F1 and ROC-AUC, plus the
    confusion matrix and a per-class classification report.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        results["roc_auc"] = roc_auc_score(y_test, y_proba)

    return results


def evaluate_multiclass_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute the multiclass-classification metric set against a held-out test set.

    Returns accuracy, macro precision, macro recall and macro F1, plus the
    confusion matrix and a per-class classification report.
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def classification_report_to_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    """Convert a sklearn classification_report dict into a tidy DataFrame."""
    return pd.DataFrame(report).transpose()
