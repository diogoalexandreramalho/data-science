"""Stage 1: preprocessing matrix at default hyperparameters.

Two public surfaces:
- `_stage_1_impl(X_train, y_train, ...)` — low-level: takes pre-loaded data
  and CV splitter, returns the result DataFrame. Used by tests and notebooks
  that already have data in memory.
- `run_stage_1(config_path)` — config-driven CLI entry point: bootstraps via
  `_context.load_context`, calls the impl, writes
  `artifacts/final/{name}/stage_1_results.csv`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import cross_validate

from data_science.experiments._context import load_context
from data_science.models.pipelines import build_pipeline


def run_stage_1(config_path: str | Path) -> pd.DataFrame:
    """Bootstrap from a config file, run the Stage 1 preprocessing matrix, save CSV."""
    ctx = load_context(config_path)
    print(f"=== {ctx.name}: Stage 1 (preprocessing matrix at defaults) ===")
    print(f"  output: {ctx.output_dir}")
    df = _stage_1_impl(
        X_train=ctx.X_train,
        y_train=ctx.y_train,
        preprocessing_configs=ctx.cfg["preprocessing"]["configs"],
        models=ctx.models,
        cv=ctx.cv,
        scoring=ctx.scoring,
        groups=ctx.groups_train,
        continuous_columns=ctx.continuous_columns,
        primary_metric=ctx.primary_metric,
    )
    out_path = ctx.output_dir / "stage_1_results.csv"
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df)} rows)")

    # Generate report comparison tables (baseline_preproc vs treatment_preproc).
    _write_report_comparison_tables(ctx, df)
    return df


def _write_report_comparison_tables(ctx, stage_1_df: pd.DataFrame) -> None:
    """Write report_table_*.csv: per-classifier primary-metric delta between
    a baseline preprocessing config and each downstream treatment config.

    Baseline is `raw` if present, else `scaled`; treatments are every other
    preprocessing config. The resulting tables match what the LaTeX report's
    §4.2.2 / §4.2.3 / §4.2.4 (and CT equivalents) tables consume.
    """
    primary_col = f"mean_{ctx.primary_metric}"
    config_names = [c["name"] for c in ctx.cfg["preprocessing"]["configs"]]

    # Pick a baseline: prefer "raw"; fall back to "scaled".
    if "raw" in config_names:
        baseline_name = "raw"
    elif "scaled" in config_names:
        baseline_name = "scaled"
    else:
        return

    baseline = stage_1_df[stage_1_df["preprocessing"] == baseline_name].set_index("model")[
        primary_col
    ]
    for treatment_name in config_names:
        if treatment_name == baseline_name:
            continue
        treatment = stage_1_df[stage_1_df["preprocessing"] == treatment_name].set_index("model")[
            primary_col
        ]
        # Reindex treatment to baseline's model order so rows align.
        treatment = treatment.reindex(baseline.index)
        table = pd.DataFrame(
            {
                "model": baseline.index,
                f"{baseline_name}_{ctx.primary_metric}": baseline.values,
                f"{treatment_name}_{ctx.primary_metric}": treatment.values,
                "difference": treatment.values - baseline.values,
            }
        )
        out = ctx.output_dir / f"report_table_{baseline_name}_vs_{treatment_name}.csv"
        table.to_csv(out, index=False)
        print(f"  wrote {out}")


def _stage_1_impl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_configs: list[dict[str, Any]],
    models: dict[str, Any],
    cv: Any,
    scoring: dict[str, str],
    groups: pd.Series | None = None,
    continuous_columns: list[str] | None = None,
    verbose: bool = True,
    primary_metric: str | None = None,
) -> pd.DataFrame:
    """Run the Stage 1 preprocessing matrix with default hyperparameters.

    For every (preprocessing config, model) pair, runs cross-validation on the
    training data and collects mean and std across folds for every metric in
    `scoring`. Returns a wide DataFrame with one row per (config, model) cell.
    """
    rows: list[dict[str, Any]] = []
    total = len(preprocessing_configs) * len(models)
    cell = 0
    stage_start = time.time()

    for config in preprocessing_configs:
        config_name = config["name"]
        for model_name, model in models.items():
            cell += 1
            cell_start = time.time()

            pipe = build_pipeline(
                X=X_train,
                model=model,
                preprocessing_config=config,
                continuous_columns=continuous_columns,
            )

            scores = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                groups=groups,
                n_jobs=-1,
                return_train_score=False,
            )

            row: dict[str, Any] = {"preprocessing": config_name, "model": model_name}
            for metric_name in scoring:
                row[f"mean_{metric_name}"] = scores[f"test_{metric_name}"].mean()
                row[f"std_{metric_name}"] = scores[f"test_{metric_name}"].std()
            rows.append(row)

            if verbose:
                elapsed_cell = time.time() - cell_start
                elapsed_total = time.time() - stage_start
                avg_per_cell = elapsed_total / cell
                eta = avg_per_cell * (total - cell)
                report_metric = primary_metric if primary_metric else next(iter(scoring))
                report_mean = row[f"mean_{report_metric}"]
                print(
                    f"  [{cell}/{total}] {config_name:35s} x {model_name:18s} "
                    f"{report_metric}={report_mean:.4f} | "
                    f"cell {elapsed_cell:5.1f}s | ETA {eta / 60:5.1f}m",
                    flush=True,
                )

    if verbose:
        print(f"  Stage 1 done in {(time.time() - stage_start) / 60:.1f} min")
    return pd.DataFrame(rows)
