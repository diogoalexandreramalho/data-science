import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    test_size: float,
    random_state: int,
    groups: pd.Series | None = None,
):
    """Create train/test splits using either grouped or stratified logic.

    For Parkinson's (strategy="group"): groups must be the patient identifier;
    all recordings from the same patient stay on the same side of the split.
    For Covertype (strategy="stratified"): the class distribution is preserved
    across train and test.
    """
    if strategy == "group":
        if groups is None:
            raise ValueError("groups must be provided for group splitting.")

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        return (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            groups.iloc[train_idx],
            groups.iloc[test_idx],
        )

    if strategy == "stratified":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        return X_train, X_test, y_train, y_test, None, None

    raise ValueError(f"Unknown split strategy: {strategy}")


def create_cv_strategy(strategy: str, n_splits: int, random_state: int):
    """Create a cross-validation splitter.

    "stratified_group_kfold" — for Parkinson's: preserves class proportions
    within folds while keeping all recordings from a patient in the same fold.
    "stratified_kfold" — for Covertype: preserves class proportions only.
    """
    if strategy == "stratified_group_kfold":
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

    if strategy == "stratified_kfold":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

    raise ValueError(f"Unknown CV strategy: {strategy}")
