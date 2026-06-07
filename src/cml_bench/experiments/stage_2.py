"""Stage 2: per-classifier grid search on the Stage-1-best preprocessing.

Public surfaces:
- `tune_classifier(...)` and `run_tune(...)` — low-level, take pre-loaded data.
- `run_stage_2(config_path)` — config-driven CLI entry point: bootstraps,
  loads `stage_1_results.csv` from disk, runs Stage 2, writes
  `artifacts/final/{name}/tuning_results.csv`.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate

from cml_bench.experiments import _wandb
from cml_bench.experiments._context import load_context, load_grids
from cml_bench.models.pipelines import build_pipeline


def run_stage_2(config_path: str | Path) -> pd.DataFrame:
    """Bootstrap, load stage_1_results.csv, run Stage 2 grid search, save CSV."""
    ctx = load_context(config_path)
    print(f"=== {ctx.name}: Stage 2 (grid search on Stage-1 winners) ===")
    print(f"  output: {ctx.output_dir}")

    stage_1_path = ctx.output_dir / "stage_1_results.csv"
    if not stage_1_path.exists():
        raise FileNotFoundError(
            f"Stage 1 results not found at {stage_1_path}. "
            f"Run `data-science stage-1 --config {config_path}` first."
        )
    stage_1_results = pd.read_csv(stage_1_path)

    grids = load_grids(ctx.source_code, list(ctx.models))
    wandb_run = _wandb.init(ctx, stage_name="stage-2")
    try:
        df = run_tune(
            stage_1_results=stage_1_results,
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            preprocessing_configs=ctx.cfg["preprocessing"]["configs"],
            models=ctx.models,
            grids=grids,
            cv=ctx.cv,
            primary_metric=ctx.primary_metric,
            groups=ctx.groups_train,
            continuous_columns=ctx.continuous_columns,
            grid_search_verbose=1,
        )
        out_path = ctx.output_dir / "tuning_results.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(df)} rows)")

        # Re-evaluate each tuned (model, preprocessing) cell with the full scoring set.
        print()
        print("  Re-evaluating each tuned classifier with the full scoring set...")
        overall = _evaluate_tuned_with_full_scoring(ctx, df)
        overall_path = ctx.output_dir / "tuned_overall_metrics.csv"
        overall.to_csv(overall_path, index=False)
        print(f"  wrote {overall_path} ({len(overall)} rows)")

        _wandb.log_dataframe(wandb_run, df, "tuning_results")
        _wandb.log_dataframe(wandb_run, overall, "tuned_overall_metrics")
        winner = df.loc[df["best_score"].idxmax()]
        _wandb.log_scalars(
            wandb_run,
            {
                f"stage_2/best_{ctx.primary_metric}": float(winner["best_score"]),
                "stage_2/winner_model": winner["model"],
                "stage_2/winner_preprocessing": winner["best_preprocessing"],
            },
        )
        return df
    finally:
        _wandb.finish(wandb_run)


def _evaluate_tuned_with_full_scoring(ctx, tuning_results: pd.DataFrame) -> pd.DataFrame:
    """For each tuned (model, preprocessing) row, run CV with the FULL scoring set."""
    configs_by_name = {c["name"]: c for c in ctx.cfg["preprocessing"]["configs"]}
    rows = []
    for _, row in tuning_results.iterrows():
        model_name = row["model"]
        preproc_name = row["best_preprocessing"]
        params = row["best_params"]
        if isinstance(params, str):
            params = ast.literal_eval(params) if params and params != "{}" else {}

        model = ctx.models[model_name]
        if params:
            model.set_params(**params)

        pipe = build_pipeline(
            X=ctx.X_train,
            model=model,
            preprocessing_config=configs_by_name[preproc_name],
            continuous_columns=ctx.continuous_columns,
        )
        cv_res = cross_validate(
            pipe,
            ctx.X_train,
            ctx.y_train,
            cv=ctx.cv,
            scoring=ctx.scoring,
            groups=ctx.groups_train,
            n_jobs=-1,
            return_train_score=False,
        )
        out_row: dict[str, Any] = {"model": model_name, "best_preprocessing": preproc_name}
        for metric_name in ctx.scoring:
            out_row[f"mean_{metric_name}"] = cv_res[f"test_{metric_name}"].mean()
            out_row[f"std_{metric_name}"] = cv_res[f"test_{metric_name}"].std()
        rows.append(out_row)
    return pd.DataFrame(rows)


def tune_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Any,
    grid: dict[str, list[Any]],
    preprocessing_config: dict[str, Any],
    cv: Any,
    primary_metric: str,
    groups: pd.Series | None = None,
    continuous_columns: list[str] | None = None,
    grid_search_verbose: int = 0,
) -> dict[str, Any]:
    """Run grid search for one (model, preprocessing) combination on training data.

    The classifier's hyperparameter grid lives in the YAML keys without prefix
    (e.g. `n_neighbors: [1, 3, 5]`); we prefix them with `model__` to address
    the model step of the sklearn Pipeline.

    `grid_search_verbose` is passed straight to `GridSearchCV.verbose`:
    0 = silent, 1 = total progress, 2 = per-combo, 3 = per-combo + score.
    """
    pipe = build_pipeline(
        X=X_train,
        model=model,
        preprocessing_config=preprocessing_config,
        continuous_columns=continuous_columns,
    )

    prefixed_grid = {f"model__{k}": v for k, v in grid.items()}

    gs = GridSearchCV(
        pipe,
        prefixed_grid,
        scoring=primary_metric,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=grid_search_verbose,
    )
    gs.fit(X_train, y_train, groups=groups)

    best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}

    return {
        "best_params": best_params,
        "best_score": gs.best_score_,
        "best_estimator": gs.best_estimator_,
    }


def run_tune(
    stage_1_results: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_configs: list[dict[str, Any]],
    models: dict[str, Any],
    grids: dict[str, dict[str, list[Any]]],
    cv: Any,
    primary_metric: str,
    groups: pd.Series | None = None,
    continuous_columns: list[str] | None = None,
    verbose: bool = True,
    grid_search_verbose: int = 0,
) -> pd.DataFrame:
    """For each classifier, tune hyperparameters on its Stage-1 winning preprocessing.

    Returns a DataFrame with one row per classifier listing the chosen
    preprocessing, the tuned hyperparameters, and the CV score of the tuned
    pipeline.
    """
    rows: list[dict[str, Any]] = []
    mean_col = f"mean_{primary_metric}"
    stage_start = time.time()

    for model_name, model in models.items():
        classifier_rows = stage_1_results[stage_1_results["model"] == model_name]
        best_row = classifier_rows.loc[classifier_rows[mean_col].idxmax()]
        best_preprocessing_name = best_row["preprocessing"]
        best_preprocessing = next(
            c for c in preprocessing_configs if c["name"] == best_preprocessing_name
        )

        grid = grids.get(model_name, {})

        if not grid:
            if verbose:
                print(
                    f"  {model_name:20s} no grid -> using Stage 1 result ({best_row[mean_col]:.4f})"
                )
            rows.append(
                {
                    "model": model_name,
                    "best_preprocessing": best_preprocessing_name,
                    "best_params": {},
                    "best_score": best_row[mean_col],
                }
            )
            continue

        grid_size = 1
        for v in grid.values():
            grid_size *= len(v)
        if verbose:
            print(
                f"  {model_name:20s} preprocessing={best_preprocessing_name}, "
                f"grid_size={grid_size} (starting...)",
                flush=True,
            )

        classifier_start = time.time()
        tune_result = tune_classifier(
            X_train=X_train,
            y_train=y_train,
            model=model,
            grid=grid,
            preprocessing_config=best_preprocessing,
            cv=cv,
            primary_metric=primary_metric,
            groups=groups,
            continuous_columns=continuous_columns,
            grid_search_verbose=grid_search_verbose,
        )
        classifier_elapsed = time.time() - classifier_start

        if verbose:
            print(
                f"  {model_name:20s} done in {classifier_elapsed / 60:5.1f}m, "
                f"best_score={tune_result['best_score']:.4f}, "
                f"best_params={tune_result['best_params']}",
                flush=True,
            )

        rows.append(
            {
                "model": model_name,
                "best_preprocessing": best_preprocessing_name,
                "best_params": tune_result["best_params"],
                "best_score": tune_result["best_score"],
            }
        )

    if verbose:
        print(f"  Stage 2 done in {(time.time() - stage_start) / 60:.1f} min")
    return pd.DataFrame(rows)
