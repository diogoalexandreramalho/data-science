"""Optional Weights & Biases experiment tracking.

Opt-in via `wandb.enabled: true` in the per-dataset YAML config (defaults to
false). When disabled, every helper here is a no-op so the pipeline runs
identically without a W&B account.

Authentication: set `WANDB_API_KEY` in the environment, or `wandb login`
before running. For local-only runs without a W&B account, set
`WANDB_MODE=offline` and W&B writes to `./wandb/` instead of the cloud.

Usage pattern in each pipeline stage:

    from cml_bench.experiments import _wandb

    run = _wandb.init(ctx, stage_name="stage-1")
    try:
        ...  # do the work
        _wandb.log_dataframe(run, df, "stage_1_results")
        _wandb.log_image(run, plot_path, "scaling_impact")
        _wandb.log_scalars(run, {"final/accuracy": 0.85})
    finally:
        _wandb.finish(run)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - wandb is in deps, but make the import optional
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


def _is_enabled(ctx) -> bool:
    cfg = ctx.cfg.get("wandb", {})
    return bool(cfg.get("enabled", False)) and _WANDB_AVAILABLE


def init(ctx, stage_name: str):
    """Initialize a W&B run for this (dataset, stage). Returns None if disabled."""
    if not _is_enabled(ctx):
        return None

    cfg = ctx.cfg.get("wandb", {})
    project = cfg.get("project", "data-science")
    entity = cfg.get("entity") or None

    return wandb.init(
        project=project,
        entity=entity,
        name=f"{ctx.name}-{stage_name}",
        group=ctx.name,
        job_type=stage_name,
        config=ctx.cfg,
        reinit=True,
    )


def finish(run) -> None:
    if run is not None:
        run.finish()


def log_scalars(run, metrics: dict[str, Any]) -> None:
    if run is None:
        return
    run.log(metrics)


def log_dataframe(run, df, key: str) -> None:
    """Log a pandas DataFrame as a wandb.Table."""
    if run is None or df is None:
        return
    run.log({key: wandb.Table(dataframe=df)})


def log_image(run, path: str | Path, key: str) -> None:
    """Log an image file (PNG/JPG) under `key`."""
    if run is None:
        return
    path = Path(path)
    if not path.exists():
        return
    run.log({key: wandb.Image(str(path))})


def log_artifact(run, path: str | Path, name: str, artifact_type: str = "model") -> None:
    """Save a file (e.g. trained model, CSV) as a versioned wandb.Artifact."""
    if run is None:
        return
    path = Path(path)
    if not path.exists():
        return
    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(str(path))
    run.log_artifact(artifact)
