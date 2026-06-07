import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


def build_preprocessor(
    X: pd.DataFrame,
    scale: bool,
    continuous_columns: list[str] | None = None,
) -> ColumnTransformer | str:
    """Build the scaling preprocessor.

    For Parkinson's (continuous_columns is None): scales every numeric column.
    For Covertype (continuous_columns is a list): scales only those columns and
    leaves the one-hot binary columns untouched.
    Returns "passthrough" when scaling is disabled.
    """
    if not scale:
        return "passthrough"

    if continuous_columns is None:
        continuous_columns = X.select_dtypes(include="number").columns.tolist()
        passthrough_columns: list[str] = []
    else:
        passthrough_columns = [col for col in X.columns if col not in continuous_columns]

    return ColumnTransformer(
        transformers=[
            ("continuous_scaler", StandardScaler(), continuous_columns),
            ("passthrough", "passthrough", passthrough_columns),
        ],
        remainder="drop",
    )


def build_feature_selector(k_best: int) -> SelectKBest:
    """Build a univariate filter feature selector using ANOVA F-scores."""
    return SelectKBest(score_func=f_classif, k=k_best)


def build_pca_transformer(n_components: float | int, random_state: int = 42) -> PCA:
    """Build a PCA transformer.

    n_components can be an integer (fixed number of components) or a float in
    (0, 1) (keep enough components to explain that fraction of variance).
    """
    return PCA(n_components=n_components, random_state=random_state)
