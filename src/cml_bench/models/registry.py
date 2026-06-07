from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_model_registry(random_state: int) -> dict[str, Any]:
    """Map model name -> sklearn estimator with default hyperparameters.

    Only `random_state` is applied here (reproducibility infrastructure, not a
    tuning hyperparameter). All other hyperparameters come from sklearn defaults
    at Stage 1 and from the grid search at Stage 2. GaussianNB and
    KNeighborsClassifier are deterministic and take no `random_state`.

    Names match the entries under `models:` in `configs/{dataset}.yaml`.
    """
    return {
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "xgboost": XGBClassifier(random_state=random_state),
    }
