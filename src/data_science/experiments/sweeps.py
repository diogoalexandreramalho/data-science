"""Preprocessing-impact + classifier-hyperparameter sweeps.

Public surfaces:
- Low-level sweep functions (used by tests / notebooks):
  `sweep_feature_selection`, `sweep_pca`, `sweep_feature_selection_per_class`,
  `sweep_classifier_hyperparameter`.
- `run_sweeps(config_path)` — config-driven CLI entry point: bootstraps,
  runs all applicable sweeps (preprocessing + classifier-hyperparameter),
  writes CSVs and plots under `artifacts/final/{name}/`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_science.evaluation.sweep_plots import (
    plot_classifier_sweep_multi_metric,
    plot_pca_variance_explained,
    plot_per_class_sweep,
    plot_scaling_impact,
    plot_sweep,
)
from data_science.experiments import _wandb
from data_science.experiments._context import load_context
from data_science.features.preprocessing import (
    build_feature_selector,
    build_pca_transformer,
    build_preprocessor,
)

# Per-dataset sweep parameters. Kept here (not in YAML) because they're tied
# to feature counts: PD has 753 features so the FS sweep goes up to 750; CT
# has 54 features so it caps at 54.
_FS_K_VALUES_BY_DATASET = {
    "parkinsons": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 300, 500, 750],
    "covertype": [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 54],
}
_PCA_N_COMPONENTS_BY_DATASET = {
    "parkinsons": [5, 10, 25, 50, 100, 150, 250, 500],
    # Covertype not included: PCA not evaluated (mixed continuous + binary).
}

# Per-classifier hyperparameter sweep specs (same for both datasets).
_CLASSIFIER_SWEEP_SPECS = [
    {
        "name": "knn",
        "title_name": "k-Nearest Neighbors",
        "classifier_cls": KNeighborsClassifier,
        "fixed": {},
        "x_param": "n_neighbors",
        "x_values": [
            1,
            3,
            5,
            7,
            9,
            11,
            13,
            15,
            17,
            19,
            21,
            25,
            31,
            35,
            40,
            45,
            50,
            52,
            55,
            60,
            70,
            80,
            90,
            100,
            105,
            110,
            120,
            130,
            140,
            150,
        ],
        "x_label": "Number of neighbors (k)",
        "line_param": "metric",
        "line_values": ["manhattan", "euclidean", "chebyshev"],
        "line_label": "Distance metric",
        "log_x": True,
    },
    {
        "name": "decision_tree",
        "title_name": "Decision Tree",
        "classifier_cls": DecisionTreeClassifier,
        "fixed": {"random_state": 42, "criterion": "entropy"},
        "x_param": "max_depth",
        "x_values": [2, 3, 4, 5, 7, 10, 25, 30, 40, 50],
        "x_label": "Maximum tree depth",
        "line_param": "min_samples_leaf",
        "line_values": [0.001, 0.005, 0.01, 0.05],
        "line_label": "min_samples_leaf",
        "log_x": False,
    },
    {
        "name": "random_forest",
        "title_name": "Random Forest",
        "classifier_cls": RandomForestClassifier,
        "fixed": {"random_state": 42, "max_features": "sqrt"},
        "x_param": "n_estimators",
        "x_values": [5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        "x_label": "Number of trees (n_estimators)",
        "line_param": "max_depth",
        "line_values": [5, 25, 70],
        "line_label": "max_depth",
        "log_x": False,
    },
    {
        "name": "gradient_boosting",
        "title_name": "Gradient Boosting",
        "classifier_cls": GradientBoostingClassifier,
        "fixed": {"random_state": 42, "learning_rate": 0.1, "max_features": "sqrt"},
        "x_param": "n_estimators",
        "x_values": [5, 10, 25, 50, 75, 100, 150, 200, 250, 300],
        "x_label": "Number of estimators",
        "line_param": "max_depth",
        "line_values": [5, 10, 25],
        "line_label": "max_depth",
        "log_x": False,
    },
    {
        "name": "xgboost",
        "title_name": "XGBoost",
        "classifier_cls": XGBClassifier,
        "fixed": {"random_state": 42},
        "x_param": "n_estimators",
        "x_values": [5, 10, 25, 50, 75, 100, 150, 200, 250, 300],
        "x_label": "Number of estimators",
        "line_param": "max_depth",
        "line_values": [5, 10, 25, 50],
        "line_label": "max_depth",
        "log_x": False,
    },
]


def run_sweeps(config_path: str | Path) -> None:
    """Bootstrap, run all applicable sweeps, save CSVs and plots."""
    ctx = load_context(config_path)
    print(f"=== {ctx.name}: sweeps ===")
    print(f"  output: {ctx.output_dir}")

    plots_dir = ctx.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()
    wandb_run = _wandb.init(ctx, stage_name="sweeps")

    # --- 1. Scaling impact (no compute — reads stage_1_results.csv) ---
    stage_1_path = ctx.output_dir / "stage_1_results.csv"
    stage_1: pd.DataFrame | None = None
    if stage_1_path.exists():
        stage_1 = pd.read_csv(stage_1_path)
        scaling_path = plots_dir / "scaling_impact.png"
        plot_scaling_impact(
            stage_1_df=stage_1,
            primary_metric=ctx.primary_metric,
            output_path=scaling_path,
            title=f"{ctx.name}: {ctx.primary_metric} by classifier under raw vs scaled",
        )
        print(f"  wrote {scaling_path}")
    else:
        print(f"  skipped scaling plot: {stage_1_path} not found")

    # --- 2. ANOVA feature ranking (cheap; ranks every feature by univariate F-statistic) ---
    print()
    print("  Computing ANOVA feature ranking...")
    _write_feature_ranking_anova(ctx)

    # --- 3. Feature selection sweep ---
    k_values = _FS_K_VALUES_BY_DATASET.get(ctx.name)
    if k_values:
        # Log scale if the k range spans more than two orders of magnitude
        # (PD: 1..750 = log; CT: 1..54 = linear). Matches the original behaviour.
        fs_log_x = max(k_values) / max(min(k_values), 1) > 100

        print()
        print(f"  Running feature selection sweep ({len(k_values)} k values)...")
        fs_df = sweep_feature_selection(
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            k_values=k_values,
            models=ctx.models,
            cv=ctx.cv,
            scoring=ctx.scoring,
            primary_metric=ctx.primary_metric,
            scale=True,
            groups=ctx.groups_train,
            continuous_columns=ctx.continuous_columns,
        )
        fs_csv = ctx.output_dir / "feature_selection_sweep.csv"
        fs_df.to_csv(fs_csv, index=False)
        plot_sweep(
            sweep_df=fs_df,
            x_col="k",
            primary_metric=ctx.primary_metric,
            output_path=plots_dir / "feature_selection_sweep.png",
            title=f"{ctx.name}: {ctx.primary_metric} vs number of selected features",
            x_label="Number of features (k)",
            log_x=fs_log_x,
        )
        print(f"  wrote {fs_csv}")

        # 2b. Per-class FS sweep
        print()
        print("  Running per-class feature selection sweep...")
        per_class_df = sweep_feature_selection_per_class(
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            k_values=k_values,
            models=ctx.models,
            cv=ctx.cv,
            class_names=ctx.class_names,
            scale=True,
            groups=ctx.groups_train,
            continuous_columns=ctx.continuous_columns,
        )
        per_class_csv = ctx.output_dir / "feature_selection_per_class.csv"
        per_class_df.to_csv(per_class_csv, index=False)
        for model_name in ctx.models:
            model_df = per_class_df[per_class_df["model"] == model_name]
            plot_per_class_sweep(
                sweep_df=model_df,
                x_col="k",
                metric="f1_score",
                output_path=plots_dir / f"feature_selection_per_class_{model_name}.png",
                title=f"{ctx.name}: per-class F1 vs k ({model_name})",
                x_label="Number of features (k)",
                log_x=fs_log_x,
            )
        print(f"  wrote {per_class_csv} + {len(ctx.models)} per-class plots")

    # --- 3. PCA variance + sweep (only if config uses PCA) ---
    uses_pca = any(cfg.get("pca", False) for cfg in ctx.cfg["preprocessing"]["configs"])
    pca_n_components = _PCA_N_COMPONENTS_BY_DATASET.get(ctx.name)
    if uses_pca and pca_n_components:
        print()
        print("  Generating PCA variance-explained plot...")
        threshold_map = plot_pca_variance_explained(
            X_train=ctx.X_train,
            output_path=plots_dir / "pca_variance_explained.png",
            title=f"{ctx.name}: cumulative variance explained vs n_components",
        )
        for t, k in threshold_map.items():
            print(f"    {int(t * 100)}% variance reached at {k} components")

        print()
        print(f"  Running PCA sweep ({len(pca_n_components)} n_components values)...")
        pca_df = sweep_pca(
            X_train=ctx.X_train,
            y_train=ctx.y_train,
            n_components_values=pca_n_components,
            models=ctx.models,
            cv=ctx.cv,
            scoring=ctx.scoring,
            primary_metric=ctx.primary_metric,
            scale=True,
            groups=ctx.groups_train,
            continuous_columns=ctx.continuous_columns,
        )
        pca_csv = ctx.output_dir / "pca_sweep.csv"
        pca_df.to_csv(pca_csv, index=False)
        plot_sweep(
            sweep_df=pca_df,
            x_col="n_components",
            primary_metric=ctx.primary_metric,
            output_path=plots_dir / "pca_sweep.png",
            title=f"{ctx.name}: {ctx.primary_metric} vs number of PCA components",
            x_label="Number of PCA components",
            log_x=True,
        )
        print(f"  wrote {pca_csv}")

    # --- 4. Per-classifier hyperparameter sweeps ---
    # Derive each classifier's Stage-1-best preprocessing from stage_1 results
    # (don't require Stage 2 tuning to have run first).
    if stage_1 is None:
        print()
        print("  Skipping classifier hyperparameter sweeps: no stage_1_results.csv.")
    else:
        configs_by_name = {c["name"]: c for c in ctx.cfg["preprocessing"]["configs"]}
        best_preproc: dict[str, dict[str, Any]] = {}
        primary_col = f"mean_{ctx.primary_metric}"
        for model_name in ctx.models:
            model_rows = stage_1[stage_1["model"] == model_name]
            if not model_rows.empty:
                best_row = model_rows.loc[model_rows[primary_col].idxmax()]
                best_preproc[model_name] = configs_by_name[best_row["preprocessing"]]

        for spec in _CLASSIFIER_SWEEP_SPECS:
            preproc_cfg = best_preproc.get(spec["name"], {"scale": True})
            print()
            print(
                f"  [{spec['title_name']}] preprocessing={preproc_cfg.get('name', 'scaled')} "
                f"sweep starting..."
            )
            sweep_df = sweep_classifier_hyperparameter(
                X_train=ctx.X_train,
                y_train=ctx.y_train,
                classifier_cls=spec["classifier_cls"],
                fixed_params=spec["fixed"],
                x_param=spec["x_param"],
                x_values=spec["x_values"],
                line_param=spec["line_param"],
                line_values=spec["line_values"],
                cv=ctx.cv,
                scoring=ctx.scoring,
                primary_metric=ctx.primary_metric,
                preprocessing_config=preproc_cfg,
                groups=ctx.groups_train,
                continuous_columns=ctx.continuous_columns,
            )
            sweep_csv = ctx.output_dir / f"classifier_sweep_{spec['name']}.csv"
            sweep_df.to_csv(sweep_csv, index=False)
            preproc_name = preproc_cfg.get("name", "scaled")
            plot_classifier_sweep_multi_metric(
                sweep_df=sweep_df,
                x_col=spec["x_param"],
                line_col=spec["line_param"],
                metrics=tuple(ctx.scoring.keys()),
                output_path=plots_dir / f"classifier_sweep_{spec['name']}.png",
                title=f"{ctx.name}: hyperparameter sweep for {spec['title_name']} ({preproc_name})",
                x_label=spec["x_label"],
                line_label=spec["line_label"],
                log_x=spec["log_x"],
            )
            print(f"    wrote {sweep_csv}")

    # W&B logging: bundle the headline sweep CSVs + key plots.
    sweep_csvs = {
        "feature_selection_sweep": ctx.output_dir / "feature_selection_sweep.csv",
        "feature_selection_per_class": ctx.output_dir / "feature_selection_per_class.csv",
        "pca_sweep": ctx.output_dir / "pca_sweep.csv",
        "feature_ranking_anova": ctx.output_dir / "feature_ranking_anova.csv",
    }
    for key, path in sweep_csvs.items():
        if path.exists():
            _wandb.log_dataframe(wandb_run, pd.read_csv(path), f"sweeps/{key}")
    for plot_name in ("scaling_impact", "feature_selection_sweep", "pca_sweep"):
        _wandb.log_image(wandb_run, plots_dir / f"{plot_name}.png", f"sweeps/plots/{plot_name}")
    for spec in _CLASSIFIER_SWEEP_SPECS:
        csv = ctx.output_dir / f"classifier_sweep_{spec['name']}.csv"
        if csv.exists():
            _wandb.log_dataframe(wandb_run, pd.read_csv(csv), f"sweeps/classifier_{spec['name']}")
            _wandb.log_image(
                wandb_run,
                plots_dir / f"classifier_sweep_{spec['name']}.png",
                f"sweeps/plots/classifier_{spec['name']}",
            )
    _wandb.finish(wandb_run)

    print()
    print(f"=== {ctx.name}: sweeps done in {(time.time() - overall_start) / 60:.1f} min ===")


def _write_feature_ranking_anova(ctx) -> None:
    """ANOVA F-statistic per feature, ranked, written to feature_ranking_anova.csv.

    Computed on the scaled training data (matches the scaling step inside the
    feature-selection sweep). Same scorer as SelectKBest(f_classif).
    """
    X_scaled = StandardScaler().fit_transform(ctx.X_train)
    f_scores, p_values = f_classif(X_scaled, ctx.y_train)
    rank = (
        pd.DataFrame({"feature": ctx.X_train.columns, "f_score": f_scores, "p_value": p_values})
        .sort_values("f_score", ascending=False)
        .reset_index(drop=True)
    )
    out = ctx.output_dir / "feature_ranking_anova.csv"
    rank.to_csv(out, index=False)
    print(f"  wrote {out} ({len(rank)} features)")


def _build_fs_pipeline(
    X: pd.DataFrame,
    model: Any,
    k: int,
    scale: bool,
    continuous_columns: list[str] | None,
) -> Pipeline:
    preprocessor = build_preprocessor(X=X, scale=scale, continuous_columns=continuous_columns)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selection", build_feature_selector(k_best=k)),
            ("model", model),
        ]
    )


def _build_pca_pipeline(
    X: pd.DataFrame,
    model: Any,
    n_components: float | int,
    scale: bool,
    continuous_columns: list[str] | None,
) -> Pipeline:
    preprocessor = build_preprocessor(X=X, scale=scale, continuous_columns=continuous_columns)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("pca", build_pca_transformer(n_components=n_components)),
            ("model", model),
        ]
    )


def sweep_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_values: list[int],
    models: dict[str, Any],
    cv: Any,
    scoring: dict[str, str],
    primary_metric: str,
    scale: bool = True,
    groups: pd.Series | None = None,
    continuous_columns: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep SelectKBest k for each (model, k) pair via 10-fold CV.

    Returns a long-format DataFrame with one row per (k, model) pair, all metrics
    as mean_X / std_X columns.
    """
    rows: list[dict[str, Any]] = []
    total = len(k_values) * len(models)
    cell = 0
    stage_start = time.time()

    for k in k_values:
        for model_name, model in models.items():
            cell += 1
            cell_start = time.time()
            pipe = _build_fs_pipeline(X_train, model, k, scale, continuous_columns)
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
            row: dict[str, Any] = {"k": k, "model": model_name}
            for metric_name in scoring:
                row[f"mean_{metric_name}"] = scores[f"test_{metric_name}"].mean()
                row[f"std_{metric_name}"] = scores[f"test_{metric_name}"].std()
            rows.append(row)

            if verbose:
                cell_t = time.time() - cell_start
                primary_mean = row[f"mean_{primary_metric}"]
                print(
                    f"  [{cell}/{total}] k={k:5d} x {model_name:18s} "
                    f"{primary_metric}={primary_mean:.4f} | cell {cell_t:5.1f}s",
                    flush=True,
                )

    if verbose:
        print(f"  feature selection sweep done in {(time.time() - stage_start) / 60:.1f} min")
    return pd.DataFrame(rows)


def sweep_pca(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_components_values: list[float | int],
    models: dict[str, Any],
    cv: Any,
    scoring: dict[str, str],
    primary_metric: str,
    scale: bool = True,
    groups: pd.Series | None = None,
    continuous_columns: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep PCA n_components for each (model, n_components) pair via 10-fold CV.

    n_components_values can be integers (fixed component count) or floats in
    (0, 1) (variance ratio to retain).
    """
    rows: list[dict[str, Any]] = []
    total = len(n_components_values) * len(models)
    cell = 0
    stage_start = time.time()

    for n in n_components_values:
        for model_name, model in models.items():
            cell += 1
            cell_start = time.time()
            pipe = _build_pca_pipeline(X_train, model, n, scale, continuous_columns)
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
            row: dict[str, Any] = {"n_components": n, "model": model_name}
            for metric_name in scoring:
                row[f"mean_{metric_name}"] = scores[f"test_{metric_name}"].mean()
                row[f"std_{metric_name}"] = scores[f"test_{metric_name}"].std()
            rows.append(row)

            if verbose:
                cell_t = time.time() - cell_start
                primary_mean = row[f"mean_{primary_metric}"]
                print(
                    f"  [{cell}/{total}] n_components={str(n):5s} x {model_name:18s} "
                    f"{primary_metric}={primary_mean:.4f} | cell {cell_t:5.1f}s",
                    flush=True,
                )

    if verbose:
        print(f"  PCA sweep done in {(time.time() - stage_start) / 60:.1f} min")
    return pd.DataFrame(rows)


def sweep_classifier_hyperparameter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    classifier_cls: type,
    fixed_params: dict[str, Any],
    x_param: str,
    x_values: list[Any],
    line_param: str,
    line_values: list[Any],
    cv: Any,
    scoring: dict[str, Any],
    primary_metric: str,
    preprocessing_config: dict[str, Any] | None = None,
    groups: pd.Series | None = None,
    continuous_columns: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep two hyperparameters of one classifier; x_param on x-axis, line_param as lines.

    Builds a full preprocessing pipeline (scaler + optional FS + optional PCA) per
    `preprocessing_config` (the same dict shape used in the per-dataset YAMLs), then
    appends the classifier instantiated with fixed_params + the swept parameters.
    Returns long-format DataFrame: one row per (line_val, x_val) with all metrics
    as mean_X / std_X columns.
    """
    cfg = preprocessing_config or {"scale": True}
    rows: list[dict[str, Any]] = []
    total = len(line_values) * len(x_values)
    cell = 0
    stage_start = time.time()

    for line_val in line_values:
        for x_val in x_values:
            cell += 1
            cell_start = time.time()
            params = {**fixed_params, x_param: x_val, line_param: line_val}
            clf = classifier_cls(**params)

            steps: list[tuple[str, Any]] = [
                (
                    "preprocessor",
                    build_preprocessor(
                        X=X_train,
                        scale=cfg.get("scale", False),
                        continuous_columns=continuous_columns,
                    ),
                ),
            ]
            if cfg.get("feature_selection", False):
                steps.append(
                    (
                        "feature_selection",
                        build_feature_selector(k_best=cfg["k_best"]),
                    )
                )
            if cfg.get("pca", False):
                steps.append(
                    (
                        "pca",
                        build_pca_transformer(n_components=cfg["pca_components"]),
                    )
                )
            steps.append(("model", clf))
            pipe = Pipeline(steps)

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
            row: dict[str, Any] = {x_param: x_val, line_param: line_val}
            for metric_name in scoring:
                row[f"mean_{metric_name}"] = scores[f"test_{metric_name}"].mean()
                row[f"std_{metric_name}"] = scores[f"test_{metric_name}"].std()
            rows.append(row)

            if verbose:
                cell_t = time.time() - cell_start
                primary_mean = row[f"mean_{primary_metric}"]
                print(
                    f"  [{cell}/{total}] {x_param}={str(x_val):8s} x "
                    f"{line_param}={str(line_val):10s} "
                    f"{primary_metric}={primary_mean:.4f} | cell {cell_t:5.1f}s",
                    flush=True,
                )

    if verbose:
        print(f"  sweep done in {(time.time() - stage_start) / 60:.1f} min")
    return pd.DataFrame(rows)


def sweep_feature_selection_per_class(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_values: list[int],
    models: dict[str, Any],
    cv: Any,
    class_names: list[str],
    scale: bool = True,
    groups: pd.Series | None = None,
    continuous_columns: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-class FS sweep using `cross_val_predict` + `classification_report`.

    Unlike `sweep_feature_selection` (which collects macro averages), this returns
    one row per (k, model, class) so we can plot per-class trends. `class_names`
    is indexed by class label (zero-indexed encoded class -> human name).

    Returns columns: k, model, class, precision, recall, f1_score, support.
    """
    rows: list[dict[str, Any]] = []
    total = len(k_values) * len(models)
    cell = 0
    stage_start = time.time()

    for k in k_values:
        for model_name, model in models.items():
            cell += 1
            cell_start = time.time()
            pipe = _build_fs_pipeline(X_train, model, k, scale, continuous_columns)
            y_pred = cross_val_predict(
                pipe,
                X_train,
                y_train,
                cv=cv,
                groups=groups,
                n_jobs=-1,
            )
            cr = classification_report(y_train, y_pred, output_dict=True, zero_division=0)
            for class_idx, class_name in enumerate(class_names):
                class_key = str(class_idx)
                if class_key in cr:
                    rows.append(
                        {
                            "k": k,
                            "model": model_name,
                            "class": class_name,
                            "precision": cr[class_key]["precision"],
                            "recall": cr[class_key]["recall"],
                            "f1_score": cr[class_key]["f1-score"],
                            "support": cr[class_key]["support"],
                        }
                    )

            if verbose:
                cell_t = time.time() - cell_start
                macro_f1 = cr.get("macro avg", {}).get("f1-score", float("nan"))
                print(
                    f"  [{cell}/{total}] k={k:3d} x {model_name:18s} "
                    f"macro_f1={macro_f1:.4f} | cell {cell_t:5.1f}s",
                    flush=True,
                )

    if verbose:
        print(f"  per-class FS sweep done in {(time.time() - stage_start) / 60:.1f} min")
    return pd.DataFrame(rows)
