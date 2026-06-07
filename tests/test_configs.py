"""Verify that the per-dataset YAML configs parse and have the schema
fields the pipeline expects.
"""

from __future__ import annotations

import pytest

from cml_bench.utils.config import load_config


@pytest.mark.parametrize("config_name", ["parkinsons.yaml", "covertype.yaml"])
def test_config_has_required_fields(repo_root, config_name):
    cfg = load_config(repo_root / "configs" / config_name)

    # Dataset block
    ds = cfg["dataset"]
    for key in ("name", "source_code", "path", "target_column", "class_names", "zero_index_labels"):
        assert key in ds, f"missing dataset.{key}"
    assert isinstance(ds["class_names"], list) and len(ds["class_names"]) >= 2

    # Split + CV + metrics + preprocessing + models
    assert cfg["split"]["strategy"] in {"group", "stratified"}
    assert cfg["cv"]["strategy"] in {"stratified_group_kfold", "stratified_kfold"}
    assert cfg["metrics"]["primary"] in cfg["metrics"]["scoring"]
    assert isinstance(cfg["preprocessing"]["configs"], list)
    assert len(cfg["preprocessing"]["configs"]) >= 1
    assert isinstance(cfg["models"], list) and len(cfg["models"]) >= 1


def test_parkinsons_uses_grouped_cv(repo_root):
    cfg = load_config(repo_root / "configs" / "parkinsons.yaml")
    assert cfg["split"]["strategy"] == "group"
    assert cfg["cv"]["strategy"] == "stratified_group_kfold"
    assert cfg["dataset"]["group_column"] == "id"
    assert cfg["dataset"]["zero_index_labels"] is False


def test_covertype_uses_stratified_cv(repo_root):
    cfg = load_config(repo_root / "configs" / "covertype.yaml")
    assert cfg["split"]["strategy"] == "stratified"
    assert cfg["cv"]["strategy"] == "stratified_kfold"
    assert cfg["dataset"]["zero_index_labels"] is True
    assert cfg.get("sampling", {}).get("balanced_subsample") is True


def test_hyperparameters_yaml_loads(repo_root):
    """The reduced grids file used by Stage 2 must load and contain both datasets."""
    grids = load_config(repo_root / "configs" / "hyperparameters.yaml")
    assert "PD" in grids and "CT" in grids
    for source in ("PD", "CT"):
        for classifier in ("knn", "decision_tree", "random_forest", "xgboost"):
            assert classifier in grids[source], f"missing {source}.{classifier}"
