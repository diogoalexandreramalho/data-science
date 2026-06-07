from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_MODEL_DISPLAY_NAMES = {
    "naive_bayes": "Naive Bayes",
    "knn": "kNN",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}


def _display_name(model_key: str) -> str:
    return _MODEL_DISPLAY_NAMES.get(model_key, model_key)


def plot_per_class_metrics(
    per_class_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    class_col: str = "cover_type",
    metric_cols: tuple[str, ...] = ("precision", "recall", "f1_score"),
) -> None:
    """Grouped bar chart of per-class metrics for one model.

    Expects a DataFrame with one row per class and columns for each metric in
    `metric_cols` plus `class_col` (the class name).
    """
    metric_display = {
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1",
        "f1": "F1",
        "support": "Support",
    }
    n_metrics = len(metric_cols)
    n_classes = len(per_class_df)
    x = np.arange(n_classes)
    width = 0.8 / n_metrics

    colors = plt.get_cmap("Set2")
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_classes), 5.5))
    for i, metric in enumerate(metric_cols):
        offset = (i - (n_metrics - 1) / 2) * width
        ax.bar(
            x + offset,
            per_class_df[metric].values,
            width,
            label=metric_display.get(metric, metric),
            color=colors(i),
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(per_class_df[class_col].values, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_sweep(
    sweep_df: pd.DataFrame,
    x_col: str,
    line_col: str,
    metric: str,
    output_path: str | Path,
    title: str,
    x_label: str | None = None,
    line_label: str | None = None,
    log_x: bool = False,
) -> None:
    """Line chart with `x_col` on x-axis and one line per value of `line_col`."""
    mean_col = f"mean_{metric}"

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.get_cmap("tab10")
    for color_idx, line_val in enumerate(sweep_df[line_col].unique()):
        sub = sweep_df[sweep_df[line_col] == line_val].sort_values(x_col)
        x_vals = sub[x_col].values
        means = sub[mean_col].values
        ax.plot(
            x_vals,
            means,
            marker="o",
            color=colors(color_idx),
            label=str(line_val),
            linewidth=2,
            markersize=7,
        )

    if log_x:
        ax.set_xscale("log")
        unique_x = sorted(set(sweep_df[x_col].unique()))
        ax.set_xticks(unique_x)
        ax.set_xticklabels(
            [
                str(int(x)) if isinstance(x, (int, float)) and float(x).is_integer() else str(x)
                for x in unique_x
            ],
            rotation=45,
            ha="right",
        )
        ax.minorticks_off()

    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(f"{metric} (10-fold CV)")
    ax.set_title(title)
    y_vals = sweep_df[mean_col].values
    y_pad = max(0.01, 0.1 * (y_vals.max() - y_vals.min()))
    ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)
    ax.legend(loc="best", title=line_label or line_col)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_sweep_multi_metric(
    sweep_df: pd.DataFrame,
    x_col: str,
    line_col: str,
    metrics: tuple[str, ...],
    output_path: str | Path,
    title: str,
    x_label: str | None = None,
    line_label: str | None = None,
    log_x: bool = False,
) -> None:
    """Grid of subplots: one panel per metric, each a line chart of
    ``mean_{metric}`` vs ``x_col`` with one line per ``line_col`` value.

    All panels share the same x-axis and the same set of line values; each
    panel y-axis is auto-zoomed to its own metric range.
    """
    metric_display = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }
    n = len(metrics)
    ncols = 3 if n >= 3 else n
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    colors = plt.get_cmap("tab10")
    line_values = list(sweep_df[line_col].unique())
    unique_x = sorted(set(sweep_df[x_col].unique()))

    handles, labels = [], []
    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        mean_col = f"mean_{metric}"
        for color_idx, line_val in enumerate(line_values):
            sub = sweep_df[sweep_df[line_col] == line_val].sort_values(x_col)
            (line,) = ax.plot(
                sub[x_col].values,
                sub[mean_col].values,
                marker="o",
                color=colors(color_idx),
                label=str(line_val),
                linewidth=1.8,
                markersize=5,
            )
            if i == 0:
                handles.append(line)
                labels.append(str(line_val))

        if log_x:
            ax.set_xscale("log")
            ax.set_xticks(unique_x)
            ax.set_xticklabels(
                [
                    str(int(x)) if isinstance(x, (int, float)) and float(x).is_integer() else str(x)
                    for x in unique_x
                ],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            ax.minorticks_off()

        y_vals = sweep_df[mean_col].values
        y_pad = max(0.01, 0.1 * (y_vals.max() - y_vals.min()))
        ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)
        ax.set_xlabel(x_label or x_col, fontsize=10)
        ax.set_ylabel(f"{metric_display.get(metric, metric)} (10-fold CV)", fontsize=10)
        ax.set_title(metric_display.get(metric, metric), fontsize=11)
        ax.grid(alpha=0.3, linestyle="--")

    # Hide any unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.00)
    fig.legend(
        handles,
        labels,
        title=line_label or line_col,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(line_values), 6),
        frameon=True,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.98))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_sweep(
    sweep_df: pd.DataFrame,
    x_col: str,
    metric: str,
    output_path: str | Path,
    title: str,
    x_label: str | None = None,
    class_col: str = "class",
    log_x: bool = False,
) -> None:
    """Line chart of a per-class metric vs `x_col` — one line per class.

    Expects the DataFrame already filtered to a single model.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.get_cmap("tab10")

    for color_idx, class_name in enumerate(sweep_df[class_col].unique()):
        sub = sweep_df[sweep_df[class_col] == class_name].sort_values(x_col)
        x_vals = sub[x_col].values
        means = sub[metric].values
        ax.plot(
            x_vals,
            means,
            marker="o",
            color=colors(color_idx),
            label=class_name,
            linewidth=2,
            markersize=7,
        )

    if log_x:
        ax.set_xscale("log")
        unique_x = sorted(set(sweep_df[x_col].unique()))
        ax.set_xticks(unique_x)
        ax.set_xticklabels(
            [str(int(x)) if float(x).is_integer() else f"{x:g}" for x in unique_x],
            rotation=45,
            ha="right",
        )
        ax.minorticks_off()

    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(f"{metric} (10-fold CV)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", title="Class")
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance_explained(
    X_train: pd.DataFrame,
    output_path: str | Path,
    title: str = "Cumulative variance explained vs number of PCA components",
    scale_first: bool = True,
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.9, 0.95, 0.99),
) -> dict[float, int]:
    """Plot cumulative explained variance against the number of PCA components.

    Computes PCA on the (optionally scaled) training data and plots the running
    sum of explained_variance_ratio_. Threshold markers show how many components
    are needed to reach each variance level. Returns the threshold-to-count map.
    """
    X = X_train.values if hasattr(X_train, "values") else X_train
    if scale_first:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=None, random_state=42)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.arange(1, len(cumvar) + 1)

    threshold_map: dict[float, int] = {}
    for t in thresholds:
        idx = int(np.argmax(cumvar >= t)) + 1
        threshold_map[t] = idx

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_components, cumvar, color="#5b8def", linewidth=2)
    ax.fill_between(n_components, 0, cumvar, color="#5b8def", alpha=0.1)

    for t, idx in threshold_map.items():
        ax.axhline(t, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.axvline(idx, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.annotate(
            f"{int(t * 100)}% → {idx} components",
            xy=(idx, t),
            xytext=(idx + len(cumvar) * 0.015, t - 0.04),
            fontsize=9,
            color="dimgray",
        )

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title(title)
    ax.set_xlim(0, len(cumvar))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return threshold_map


def plot_scaling_impact(
    stage_1_df: pd.DataFrame,
    primary_metric: str,
    output_path: str | Path,
    title: str = "Effect of scaling on the primary metric",
    raw_config_name: str = "raw",
    scaled_config_name: str = "scaled",
) -> None:
    """Bar chart: primary metric per classifier under raw vs scaled preprocessing.

    Reads the Stage 1 results DataFrame and plots paired bars for the two
    preprocessing configurations specified by `raw_config_name` and
    `scaled_config_name`.
    """
    mean_col = f"mean_{primary_metric}"
    std_col = f"std_{primary_metric}"

    raw_rows = stage_1_df[stage_1_df["preprocessing"] == raw_config_name]
    scaled_rows = stage_1_df[stage_1_df["preprocessing"] == scaled_config_name]

    models = list(raw_rows["model"].values)
    raw_means = raw_rows.set_index("model")[mean_col].reindex(models).values
    raw_stds = raw_rows.set_index("model")[std_col].reindex(models).values
    scaled_means = scaled_rows.set_index("model")[mean_col].reindex(models).values
    scaled_stds = scaled_rows.set_index("model")[std_col].reindex(models).values

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(models)), 5))
    ax.bar(
        x - width / 2,
        raw_means,
        width,
        yerr=raw_stds,
        capsize=4,
        label="Raw",
        color="#5b8def",
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        scaled_means,
        width,
        yerr=scaled_stds,
        capsize=4,
        label="Scaled",
        color="#f29e4c",
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([_display_name(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel(f"Mean {primary_metric} (10-fold CV)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(
    sweep_df: pd.DataFrame,
    x_col: str,
    primary_metric: str,
    output_path: str | Path,
    title: str,
    x_label: str | None = None,
    log_x: bool = False,
    baseline: float | None = None,
    baseline_label: str | None = None,
) -> None:
    """Line chart of primary metric (mean across CV folds) vs `x_col`, one line per model.

    If `baseline` is given, draws a horizontal reference line (e.g. majority-class
    F1 = 0.854, or random ROC-AUC = 0.5).
    """
    mean_col = f"mean_{primary_metric}"

    fig, ax = plt.subplots(figsize=(11, 6))

    if baseline is not None:
        label = baseline_label or f"Baseline = {baseline:.3f}"
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2, alpha=0.5, label=label)

    colors = plt.get_cmap("tab10")
    for color_idx, model_name in enumerate(sweep_df["model"].unique()):
        sub = sweep_df[sweep_df["model"] == model_name].sort_values(x_col)
        x_vals = sub[x_col].values
        means = sub[mean_col].values
        color = colors(color_idx)
        ax.plot(
            x_vals,
            means,
            marker="o",
            color=color,
            label=_display_name(model_name),
            linewidth=2,
            markersize=7,
        )

    if log_x:
        ax.set_xscale("log")
        # Label exactly the measured x values; turn off minor ticks to avoid clutter.
        unique_x = sorted(set(sweep_df[x_col].unique()))
        ax.set_xticks(unique_x)
        ax.set_xticklabels(
            [str(int(x)) if float(x).is_integer() else f"{x:g}" for x in unique_x],
            rotation=45,
            ha="right",
        )
        ax.minorticks_off()

    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(f"Mean {primary_metric} (10-fold CV)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
