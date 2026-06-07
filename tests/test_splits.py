"""Critical invariant: patient-grouped split must NOT leak patients across
train/test. This protects against the dominant data-leakage failure mode on
the Parkinson's dataset (multiple recordings per patient).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_science.data.split import create_cv_strategy, create_train_test_split


def _synthetic_patient_recordings(
    n_patients: int = 12, recordings_per_patient: int = 3, seed: int = 0
):
    """Build a synthetic frame mimicking PD's structure: multiple recordings per patient
    with a single label per patient."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient_id in range(n_patients):
        label = int(patient_id < n_patients * 0.75)  # 75/25 imbalanced, mimicking PD
        for _ in range(recordings_per_patient):
            rows.append(
                {
                    "patient_id": patient_id,
                    "feat_1": rng.standard_normal(),
                    "feat_2": rng.standard_normal(),
                    "label": label,
                }
            )
    df = pd.DataFrame(rows)
    X = df[["feat_1", "feat_2"]]
    y = df["label"]
    groups = df["patient_id"]
    return X, y, groups


def test_group_split_does_not_leak_patients():
    X, y, groups = _synthetic_patient_recordings()
    _, _, _, _, g_train, g_test = create_train_test_split(
        X,
        y,
        strategy="group",
        test_size=0.25,
        random_state=42,
        groups=groups,
    )
    assert set(g_train).isdisjoint(set(g_test)), f"patient leakage: {set(g_train) & set(g_test)}"


def test_group_cv_folds_have_no_patient_leakage():
    """Every fold of stratified_group_kfold must keep patients disjoint."""
    X, y, groups = _synthetic_patient_recordings()
    cv = create_cv_strategy(strategy="stratified_group_kfold", n_splits=3, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])
        assert train_groups.isdisjoint(val_groups), (
            f"fold {fold_idx}: leakage {train_groups & val_groups}"
        )


def test_stratified_split_preserves_class_balance():
    """Stratified (non-group) split should keep class proportions close to original."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feat_1": rng.standard_normal(200),
            "feat_2": rng.standard_normal(200),
            "label": rng.choice([0, 1, 2], size=200, p=[0.5, 0.3, 0.2]),
        }
    )
    X = df[["feat_1", "feat_2"]]
    y = df["label"]
    _, _, y_train, y_test, _, _ = create_train_test_split(
        X,
        y,
        strategy="stratified",
        test_size=0.25,
        random_state=42,
    )
    # Each class should appear in both train and test
    assert set(y_train.unique()) == set(y_test.unique()) == {0, 1, 2}
