import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data_science.datasets import DATASETS
from data_science.models import decision_tree as dt
from data_science.models import gradient_boost as gb
from data_science.models import knn
from data_science.models import naive_bayes as nb
from data_science.models import random_forest as rf
from data_science.models import xgboost_clf as xgb
from data_science.preprocessing import data_balancing as balance
from data_science.preprocessing import normalize as norm
from data_science.viz import print_statistics as stats


def classification(data, source):
    target = DATASETS[source].target_column

    # split data set in target variable and atributes
    data = data.apply(pd.to_numeric)
    y: np.ndarray = data.pop(target).values.astype(int)
    X: np.ndarray = data.values.astype(float)
    labels: np.ndarray = pd.unique(y)

    # store accuracies and specificities for each classifier
    accuracies = {"nb": [], "knn": [], "dt": [], "rf": [], "gb": [], "xgb": []}
    if source == "PD":
        specificities = {"nb": [], "knn": [], "dt": [], "rf": [], "gb": [], "xgb": []}

    n_classes = len(labels)
    cnf_mtxs_temp = [np.zeros((n_classes, n_classes), dtype=int) for _ in range(6)]
    cnf_mtxs = [np.zeros((n_classes, n_classes), dtype=int) for _ in range(6)]

    cv = KFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in cv.split(X):
        trnX, tstX, trnY, tstY = X[train_index], X[test_index], y[train_index], y[test_index]

        # normalize and balance the dataset
        trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
        trnX, trnY = balance.run(trnX, trnY, "all", 42, False)

        # classify with fixed parameters and get the metrics
        acc = [0, 0, 0, 0, 0, 0]

        if source == "PD":
            spec = [0, 0, 0, 0, 0, 0]

            acc[0], spec[0], cnf_mtxs_temp[0] = nb.simple_naive_bayes(
                trnX, tstX, trnY, tstY, labels
            )
            acc[1], spec[1], cnf_mtxs_temp[1] = knn.simple_knn(
                trnX, tstX, trnY, tstY, 1, "manhattan", labels
            )
            acc[2], spec[2], cnf_mtxs_temp[2] = dt.simple_decision_tree(
                trnX, tstX, trnY, tstY, 0.05, 5, "entropy", labels
            )
            acc[3], spec[3], cnf_mtxs_temp[3] = rf.simple_random_forest(
                trnX, tstX, trnY, tstY, 150, 10, "sqrt", labels
            )
            acc[4], spec[4], cnf_mtxs_temp[4] = gb.simple_gradient_boost(
                trnX, tstX, trnY, tstY, 100, 0.1, 5, "sqrt", labels
            )
            acc[5], spec[5], cnf_mtxs_temp[5] = xgb.simple_xg_boost(
                trnX, tstX, trnY, tstY, 200, 5, labels
            )
        else:
            acc[0], cnf_mtxs_temp[0] = nb.simple_naive_bayes_CT(trnX, tstX, trnY, tstY, labels)
            acc[1], cnf_mtxs_temp[1] = knn.simple_knn_CT(
                trnX, tstX, trnY, tstY, 1, "manhattan", labels
            )
            acc[2], cnf_mtxs_temp[2] = dt.simple_decision_tree_CT(
                trnX, tstX, trnY, tstY, 0.05, 5, "entropy", labels
            )
            acc[3], cnf_mtxs_temp[3] = rf.simple_random_forest_CT(
                trnX, tstX, trnY, tstY, 150, 10, "sqrt", labels
            )
            acc[4], cnf_mtxs_temp[4] = gb.simple_gradient_boost_CT(
                trnX, tstX, trnY, tstY, 100, 0.1, 5, "sqrt", labels
            )
            acc[5], cnf_mtxs_temp[5] = xgb.simple_xg_boost_CT(
                trnX, tstX, trnY, tstY, 200, 5, labels
            )

        # add confusion matrixes
        for i in range(6):
            cnf_mtxs[i] = np.add(cnf_mtxs[i], cnf_mtxs_temp[i])

        # store metrics
        for i, clf in enumerate(accuracies):
            accuracies[clf].append(acc[i])
            if source == "PD":
                specificities[clf].append(spec[i])

    # calculate avg accuracy and avg specificity
    avg_accuracies = []
    avg_specificities = []
    for clf in accuracies:
        avg_acc_clf = sum(accuracies[clf]) / len(accuracies[clf])
        avg_accuracies += [avg_acc_clf]
        if source == "PD":
            avg_spec_clf = sum(specificities[clf]) / len(specificities[clf])
            avg_specificities += [avg_spec_clf]

    clf_names = [
        "Naive Bayes",
        "kNN",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
    ]

    # create report for each classifier
    if source == "PD":
        params = [
            ("GaussianNB"),
            ("manhattan", 1),
            ("entropy", 5, 0.05),
            ("sqrt", 10, 150),
            ("sqrt", 5, 100, 0.1),
            (5, 200),
        ]

        nb_report = [clf_names[0], params[0], avg_accuracies[0], avg_specificities[0], cnf_mtxs[0]]
        knn_report = [clf_names[1], params[1], avg_accuracies[1], avg_specificities[1], cnf_mtxs[1]]
        dt_report = [clf_names[2], params[2], avg_accuracies[2], avg_specificities[2], cnf_mtxs[2]]
        rf_report = [clf_names[3], params[3], avg_accuracies[3], avg_specificities[3], cnf_mtxs[3]]
        gb_report = [clf_names[4], params[4], avg_accuracies[4], avg_specificities[4], cnf_mtxs[4]]
        xgb_report = [clf_names[5], params[5], avg_accuracies[5], avg_specificities[5], cnf_mtxs[5]]

        reports = [nb_report, knn_report, dt_report, rf_report, gb_report, xgb_report]

        stats.print_report(reports, (True, True))
    else:
        params = [
            ("GaussianNB"),
            ("manhattan", 1),
            ("entropy", 50, 0.00005),
            ("sqrt", 25, 185),
            ("sqrt", 10, 300, 0.05),
            (10, 300),
        ]

        nb_report = [clf_names[0], [params[0], avg_accuracies[0], cnf_mtxs[0]]]
        knn_report = [clf_names[1], [params[1], avg_accuracies[1], cnf_mtxs[1]]]
        dt_report = [clf_names[2], [params[2], avg_accuracies[2], cnf_mtxs[2]]]
        rf_report = [clf_names[3], [params[3], avg_accuracies[3], cnf_mtxs[3]]]
        gb_report = [clf_names[4], [params[4], avg_accuracies[4], cnf_mtxs[4]]]
        xgb_report = [clf_names[5], [params[5], avg_accuracies[5], cnf_mtxs[5]]]

        reports = [nb_report, knn_report, dt_report, rf_report, gb_report, xgb_report]

        stats.print_analysis_CT(reports, (True, True))
