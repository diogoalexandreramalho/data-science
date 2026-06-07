"""Stage 3: refit Stage-2 winner on full training set, evaluate on held-out test.

Public surfaces:
- `run_final(tuning_results, X_train, ...)` — low-level, takes pre-loaded data.
- `run_final_evaluation(config_path)` — config-driven CLI entry point:
  bootstraps, loads `tuning_results.csv` from disk, runs final eval, writes
  `artifacts/final/{name}/final_metrics.json` + `confusion_matrix.png` +
  (multiclass only) `per_class_metrics.csv`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_science.evaluation.metrics import (
    evaluate_binary_classifier,
    evaluate_multiclass_classifier,
)
from data_science.evaluation.plots import plot_confusion_matrix
from data_science.experiments._context import load_context
from data_science.models.pipelines import build_pipeline


def _parse_params(param_str: Any) -> dict[str, Any]:
    """tuning_results.csv stores best_params as a Python-repr string; parse back to dict."""
    if isinstance(param_str, dict):
        return param_str
    if not param_str or param_str == "{}" or pd.isna(param_str):
        return {}
    return ast.literal_eval(param_str)


def run_final_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Bootstrap, load tuning_results.csv, refit winner, evaluate on test, save outputs."""
    ctx = load_context(config_path)
    print(f"=== {ctx.name}: Stage 3 (final on held-out test) ===")
    print(f"  output: {ctx.output_dir}")

    tuning_path = ctx.output_dir / "tuning_results.csv"
    if not tuning_path.exists():
        raise FileNotFoundError(
            f"Tuning results not found at {tuning_path}. "
            f"Run `data-science stage-2 --config {config_path}` first."
        )
    tuning_results = pd.read_csv(tuning_path)
    tuning_results["best_params"] = tuning_results["best_params"].apply(_parse_params)

    confusion_path = ctx.output_dir / "confusion_matrix.png"
    result = run_final(
        tuning_results=tuning_results,
        X_train=ctx.X_train,
        y_train=ctx.y_train,
        X_test=ctx.X_test,
        y_test=ctx.y_test,
        preprocessing_configs=ctx.cfg["preprocessing"]["configs"],
        models=ctx.models,
        is_binary=ctx.is_binary,
        class_names=ctx.class_names,
        confusion_matrix_path=confusion_path,
        continuous_columns=ctx.continuous_columns,
        confusion_matrix_title=f"{ctx.name} best model — held-out test",
    )

    classification_report = result["metrics"]["classification_report"]
    final_metrics = {
        "best_model": result["best_model"],
        "best_preprocessing": result["best_preprocessing"],
        "best_params": result["best_params"],
        "test_metrics": {
            k: v for k, v in result["metrics"].items() if k != "classification_report"
        },
        "classification_report": classification_report,
    }
    final_path = ctx.output_dir / "final_metrics.json"
    final_path.write_text(json.dumps(final_metrics, indent=2, default=str))
    print(f"  wrote {final_path}")
    print(f"  wrote {confusion_path}")

    if not ctx.is_binary:
        per_class_rows = []
        for encoded_label, class_name in enumerate(ctx.class_names):
            if str(encoded_label) in classification_report:
                cls_metrics = classification_report[str(encoded_label)]
                per_class_rows.append(
                    {
                        "class": class_name,
                        "precision": cls_metrics["precision"],
                        "recall": cls_metrics["recall"],
                        "f1_score": cls_metrics["f1-score"],
                        "support": cls_metrics["support"],
                    }
                )
        per_class_path = ctx.output_dir / "per_class_metrics.csv"
        pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
        print(f"  wrote {per_class_path}")

    print(f"  best: {result['best_model']} ({result['best_preprocessing']})")
    print(f"  test accuracy: {result['metrics']['accuracy']:.4f}")
    primary_test_key = ctx.primary_metric if ctx.primary_metric in result["metrics"] else None
    if primary_test_key:
        print(f"  test {primary_test_key}: {result['metrics'][primary_test_key]:.4f}")
    return result


def run_final(
    tuning_results: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessing_configs: list[dict[str, Any]],
    models: dict[str, Any],
    is_binary: bool,
    class_names: list[str],
    confusion_matrix_path: str | Path,
    continuous_columns: list[str] | None = None,
    confusion_matrix_title: str = "Confusion matrix",
) -> dict[str, Any]:
    """Pick the overall best (model, preprocessing) from tuning, refit on the full
    training set, evaluate on the held-out test set, and save the confusion matrix.
    """
    winner_row = tuning_results.loc[tuning_results["best_score"].idxmax()]
    winner_model_name = winner_row["model"]
    winner_preprocessing_name = winner_row["best_preprocessing"]
    winner_params = winner_row["best_params"]

    winner_preprocessing = next(
        c for c in preprocessing_configs if c["name"] == winner_preprocessing_name
    )

    model = models[winner_model_name]
    if winner_params:
        model.set_params(**winner_params)

    pipe = build_pipeline(
        X=X_train,
        model=model,
        preprocessing_config=winner_preprocessing,
        continuous_columns=continuous_columns,
    )
    pipe.fit(X_train, y_train)

    if is_binary:
        metrics = evaluate_binary_classifier(pipe, X_test, y_test)
    else:
        metrics = evaluate_multiclass_classifier(pipe, X_test, y_test)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=class_names,
        output_path=confusion_matrix_path,
        title=confusion_matrix_title,
    )

    return {
        "best_model": winner_model_name,
        "best_preprocessing": winner_preprocessing_name,
        "best_params": winner_params,
        "metrics": metrics,
    }
