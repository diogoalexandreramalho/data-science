"""Training and hyperparameter-tuning utilities for sklearn estimators."""

import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV


def train(estimator, trnX, tstX, trnY, tstY, labels):
    """Fit one estimator instance, return (accuracy, recall_or_None, confusion_matrix).

    Recall is computed only for binary classification (len(labels) == 2);
    multi-class returns None for that slot.
    """
    estimator.fit(trnX, trnY)
    prdY = estimator.predict(tstX)
    accuracy = metrics.accuracy_score(tstY, prdY)
    cnf_matrix = metrics.confusion_matrix(tstY, prdY, labels=labels)
    recall = None
    if len(labels) == 2:
        recall = metrics.recall_score(tstY, prdY)
    return accuracy, recall, cnf_matrix


def sweep(estimator_cls, X, y, param_grid, scoring="accuracy", cv=10):
    """Grid-search hyperparameters via cross-validated GridSearchCV.

    Returns the fitted GridSearchCV object — has .best_params_, .best_score_, .cv_results_.
    """
    grid = GridSearchCV(estimator_cls(), param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid.fit(X, y)
    return grid
