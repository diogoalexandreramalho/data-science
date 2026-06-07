import pandas as pd


def create_balanced_sample(
    df: pd.DataFrame,
    target_column: str,
    samples_per_class: int,
    random_state: int,
) -> pd.DataFrame:
    """Create a balanced sample with the same number of rows per class.

    Used for the Covertype dataset, where the original class distribution is
    highly imbalanced and must be reduced to a tractable size for repeated
    cross-validation across multiple classifiers.
    """
    sampled_parts = []

    for class_value, class_df in df.groupby(target_column):
        if len(class_df) < samples_per_class:
            raise ValueError(
                f"Class {class_value} has only {len(class_df)} rows, "
                f"but {samples_per_class} were requested."
            )

        sampled_parts.append(
            class_df.sample(
                n=samples_per_class,
                random_state=random_state,
            )
        )

    return (
        pd.concat(sampled_parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
