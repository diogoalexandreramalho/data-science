"""Verify that all artifacts the LaTeX report references exist on disk.

Each test SKIPS (does not fail) when the artifact is missing — so a fresh
clone with no `make reproduce` run shows the test as skipped, not failed.
After `make reproduce`, all of these should pass.
"""

from __future__ import annotations

import pytest

REQUIRED_BY_DATASET = {
    "parkinsons": [
        "stage_1_results.csv",
        "tuning_results.csv",
        "final_metrics.json",
        "confusion_matrix.png",
        "feature_selection_sweep.csv",
        "feature_selection_per_class.csv",
        "pca_sweep.csv",
        "classifier_sweep_knn.csv",
        "classifier_sweep_decision_tree.csv",
        "classifier_sweep_random_forest.csv",
        "classifier_sweep_gradient_boosting.csv",
        "classifier_sweep_xgboost.csv",
        "plots/scaling_impact.png",
        "plots/feature_selection_sweep.png",
        "plots/pca_sweep.png",
    ],
    "covertype": [
        "stage_1_results.csv",
        "tuning_results.csv",
        "final_metrics.json",
        "confusion_matrix.png",
        "per_class_metrics.csv",
        "feature_selection_sweep.csv",
        "feature_selection_per_class.csv",
        "classifier_sweep_knn.csv",
        "classifier_sweep_decision_tree.csv",
        "classifier_sweep_random_forest.csv",
        "classifier_sweep_gradient_boosting.csv",
        "classifier_sweep_xgboost.csv",
        "plots/scaling_impact.png",
        "plots/feature_selection_sweep.png",
    ],
}


@pytest.mark.parametrize(
    "dataset,relative_path",
    [(ds, rel) for ds, paths in REQUIRED_BY_DATASET.items() for rel in paths],
)
def test_required_artifact_exists(repo_root, dataset, relative_path):
    path = repo_root / "artifacts" / "final" / dataset / relative_path
    if not path.exists():
        pytest.skip(f"{path.relative_to(repo_root)} not present — run `make reproduce` first")
    assert path.stat().st_size > 0, f"{path.relative_to(repo_root)} is empty"


def test_report_pdf_exists(repo_root):
    pdf = repo_root / "reports" / "report.pdf"
    if not pdf.exists():
        pytest.skip("reports/report.pdf not present — run `make report` first")
    assert pdf.stat().st_size > 100_000, "report.pdf suspiciously small"
