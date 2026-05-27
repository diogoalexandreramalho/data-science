import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    f_classif,
)
from sklearn.model_selection import train_test_split

from data_science.pipeline import get_classifier
from data_science.preprocessing import data_balancing as balance
from data_science.preprocessing import normalize as norm
from data_science.train import train

#
#
#                       A N A L Y S I S
#
#


def balance_analysis(dataset, name):
    data = dataset.copy()
    # Prepare data
    if name == "PD":  # Pd_dataset
        data["class"]  # Save the class for later
        data.pop("id")
        y = data.pop("class").values
        X = data.values

    else:  # Cover_Type
        data = data.groupby("Cover_Type").apply(lambda s: s.sample(100))
        y: np.ndarray = data.pop("Cover_Type").values
        X: np.ndarray = data.values

    classifiers = ["Naive_Bayes", "KNN", "Decision_Tree"]
    balance = ["SMOTE", "None"]
    accuracy = []
    specificity = []

    labels: np.ndarray = pd.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=41
    )

    print("Balance Performance")
    for clf in classifiers:
        for bal in balance:
            if name == "PD":
                acc, spec, conf = classify_balance(
                    X_train, X_test, y_train, y_test, labels, clf, bal, name
                )
                accuracy.append(acc)
                specificity.append(spec)

            else:
                acc, conf = classify_balance(
                    X_train, X_test, y_train, y_test, labels, clf, bal, name
                )
                accuracy.append(acc)

        performance_balance(accuracy, specificity, clf, name)
        accuracy.clear()
        specificity.clear()


def normalize_analysis(dataset, name):
    data = dataset.copy()
    # Prepare data
    if name == "PD":  # Pd_dataset
        data["class"]  # Save the class for later
        data.pop("id")
        y = data.pop("class").values
        X = data.values

    else:  # Cover_Type
        data = data.groupby("Cover_Type").apply(lambda s: s.sample(100))
        y: np.ndarray = data.pop("Cover_Type").values
        X: np.ndarray = data.values

    classifiers = ["Naive_Bayes", "KNN", "Decision_Tree"]
    normalizers = ["min_max_scaler", "standard_scaler", "None"]
    accuracy = []
    specificity = []

    labels: np.ndarray = pd.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=41
    )

    print("Normalization Performance")
    for clf in classifiers:
        for n in normalizers:
            if name == "PD":
                acc, spec, conf = classify_normalization(
                    X_train, X_test, y_train, y_test, labels, clf, n, name
                )
                accuracy.append(acc)
                specificity.append(spec)

            else:
                acc, conf = classify_normalization(
                    X_train, X_test, y_train, y_test, labels, clf, n, name
                )
                accuracy.append(acc)

        performance_normalization(accuracy, specificity, clf, name)
        accuracy.clear()
        specificity.clear()


def feature_selection_analysis(dataset, name):
    data = dataset.copy()
    # Prepare data
    if name == "PD":  # Pd_dataset
        data["class"]  # Save the class for later
        data.pop("id")
        y = data.pop("class").values
        X = data.values

    else:  # Cover_Type
        warnings.filterwarnings("ignore")
        data = data.groupby("Cover_Type").apply(lambda s: s.sample(100))
        y: np.ndarray = data.pop("Cover_Type").values
        X: np.ndarray = data.values

    # This computes fval, and pval for every feature
    # fval, pval = f_classif(X, y)

    classifiers = ["Naive_Bayes", "KNN"]
    accuracy = []
    specificity = []
    result_features = {"SelectKBest": [0, 0], "SelectPercentile": [0, 0], "Wrapper": [0, 0]}

    result_perf = {"SelectKBest": [0, 0], "SelectPercentile": [0, 0], "Wrapper": [0, 0]}

    #
    #                      S E L E C T  K  B E S T
    #

    arg = math.floor(math.log(X.shape[1], 2))
    n_features = 2 ** np.arange(arg + 1)  # [1,2,4,8,16 ...]
    n_features = np.append(n_features, [X.shape[1]], axis=0)

    # Computing sequential foward selections for classifier
    print("Performance of SelectKBest")
    for clf in classifiers:
        for k in n_features:
            selector = SelectKBest(f_classif, k=k)  # k=2 means, I only pick 2 best features
            X_new = selector.fit_transform(X, y)

            # apply classifier -> Returns accuracy and specificity
            if name == "PD":
                acc, spec, conf = classify_feature_selection(X_new, y, clf, name)
                accuracy.append(acc)
                specificity.append(spec)

            else:
                acc, conf = classify_feature_selection(X_new, y, clf, name)
                accuracy.append(acc)

        # plot performance graphs
        # plot_feature_selection(n_features, accuracy, specificity, clf, "SelectKBest")
        performance_feature_selection(
            accuracy,
            specificity,
            n_features,
            clf,
            "SelectKBest",
            result_features,
            result_perf,
            name,
        )

        accuracy.clear()
        specificity.clear()

    #
    #                      S E L E C T   P E R C E N T I L E
    #

    percentile = np.arange(1, 11) * 10
    accuracy.clear()
    specificity.clear()

    print("Performance of SelectPercentile")
    for clf in classifiers:
        for perc in percentile:
            selector = SelectPercentile(f_classif, percentile=perc)
            X_new = selector.fit_transform(X, y)

            # apply classifier -> Returns accuracy and specificity
            if name == "PD":
                acc, spec, conf = classify_feature_selection(X_new, y, clf, name)
                accuracy.append(acc)
                specificity.append(spec)

            else:
                acc, conf = classify_feature_selection(X_new, y, clf, name)
                accuracy.append(acc)

        # plot performance graphs
        # plot_feature_selection(percentile, accuracy, specificity, clf, "SelectPercentile")
        performance_feature_selection(
            accuracy,
            specificity,
            percentile,
            clf,
            "SelectPercentile",
            result_features,
            result_perf,
            name,
        )

        accuracy.clear()
        specificity.clear()

    #
    #                           M O D E L   B A S E D
    #

    # Phase - 1
    clf1 = ExtraTreesClassifier(n_estimators=50)
    clf1 = clf1.fit(X, y)
    model = SelectFromModel(clf1, prefit=True)
    X_new = model.transform(X)  # -> Chooses the best features

    number_features = X_new.shape[1]

    print("Performance of Wrapper")
    for clf in classifiers:
        # Phase - 2
        # apply classifier -> Returns accuracy and specificity
        if name == "PD":
            acc, spec, conf = classify_feature_selection(X_new, y, clf, name)
            accuracy.append(acc)
            specificity.append(spec)

        else:
            acc, conf = classify_feature_selection(X_new, y, clf, name)
            accuracy.append(acc)

        performance_feature_selection(
            accuracy,
            specificity,
            number_features,
            clf,
            "Wrapper",
            result_features,
            result_perf,
            name,
        )

    # Average of features according with bayes and knn
    for feat in result_features.values():
        feat[0] = feat[0] // 2
        feat[1] = feat[1] // 2

    for perf in result_perf.values():
        perf[0] = perf[0] / 2
        perf[1] = perf[1] / 2

    compareFeatures(result_features)
    comparePerformance(result_perf, name)

    return


def pca_analysis(dataset, name):
    data = dataset.copy()

    if name == "PD":
        clss = data.pop("class")
    else:
        clss = data.pop("Cover_Type")

    if name == "PD":
        n_components = np.arange(100, data.shape[0], 50)

    else:
        n_components = np.arange(10, min(data.shape[0], data.shape[1]), 5)

    classifiers = ["Naive_Bayes", "KNN"]
    accuracy = []
    specificity = []

    # PCA
    print("PCA Performance")
    for clf in classifiers:
        for comp in n_components:
            pca = PCA(n_components=comp, svd_solver="full")
            pca.fit_transform(data)

            new_data = pd.DataFrame(pca.components_, columns=data.columns)
            if name == "PD":
                new_data["class"] = clss
            else:
                new_data["Cover_Type"] = clss

            # apply classifier -> Returns accuracy and specificity
            if name == "PD":
                acc, spec, conf = classify_pca(new_data, clf, name)
                accuracy.append(acc)
                specificity.append(spec)

            else:
                acc, conf = classify_pca(new_data, clf, name)
                accuracy.append(acc)

        performance_pca(accuracy, specificity, n_components, clf, name)
        accuracy.clear()
        specificity.clear()


#
#
#                       C L A S S I F Y
#
#


def classify_balance(X_train, X_test, y_train, y_test, labels, clf, bal, name):

    X_train, X_test, y_train, y_test = norm.standard_scaler(X_train, X_test, y_train, y_test)

    if bal == "SMOTE":
        X_train, y_train = balance.run(X_train, y_train, "all", 42, False)

    elif bal == "None":
        # Does nothing
        pass

    # classify
    if clf == "Naive_Bayes":
        if name == "PD":
            return train(
                get_classifier("naive_bayes", name), X_train, X_test, y_train, y_test, labels
            )
        else:
            return train(
                get_classifier("naive_bayes", name), X_train, X_test, y_train, y_test, labels
            )

    elif clf == "KNN":
        if name == "PD":
            return train(get_classifier("knn", name), X_train, X_test, y_train, y_test, labels)
        else:
            return train(get_classifier("knn", name), X_train, X_test, y_train, y_test, labels)

    elif clf == "Decision_Tree":
        if name == "PD":
            return train(
                get_classifier("decision_tree", name), X_train, X_test, y_train, y_test, labels
            )
        else:
            return train(
                get_classifier("decision_tree", name), X_train, X_test, y_train, y_test, labels
            )


def classify_normalization(X_train, X_test, y_train, y_test, labels, clf, n, name):

    if n == "min_max_scaler":
        X_train, X_test, y_train, y_test = norm.min_max_scaler(X_train, X_test, y_train, y_test)

    elif n == "standard_scaler":
        X_train, X_test, y_train, y_test = norm.standard_scaler(X_train, X_test, y_train, y_test)

    elif n == "None":
        # Does nothing
        pass

    X_train, y_train = balance.run(X_train, y_train, "all", 42, False)

    # classify
    if clf == "Naive_Bayes":
        if name == "PD":
            return train(
                get_classifier("naive_bayes", name), X_train, X_test, y_train, y_test, labels
            )
        else:
            return train(
                get_classifier("naive_bayes", name), X_train, X_test, y_train, y_test, labels
            )

    elif clf == "KNN":
        if name == "PD":
            return train(get_classifier("knn", name), X_train, X_test, y_train, y_test, labels)
        else:
            return train(get_classifier("knn", name), X_train, X_test, y_train, y_test, labels)

    elif clf == "Decision_Tree":
        if name == "PD":
            return train(
                get_classifier("decision_tree", name), X_train, X_test, y_train, y_test, labels
            )
        else:
            return train(
                get_classifier("decision_tree", name), X_train, X_test, y_train, y_test, labels
            )


def classify_feature_selection(X, y, clf, name):

    labels: np.ndarray = pd.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=41
    )

    # normalize and balance the dataset
    X_train, X_test, y_train, y_test = norm.standard_scaler(X_train, X_test, y_train, y_test)
    X_train, y_train = balance.run(X_train, y_train, "all", 42, False)

    if clf == "Naive_Bayes":
        if name == "PD":
            return train(
                get_classifier("naive_bayes", name), X_train, X_test, y_train, y_test, labels
            )
        else:
            return train(
                get_classifier("naive_bayes", name), X_train, X_test, y_train, y_test, labels
            )

    elif clf == "KNN":
        if name == "PD":
            return train(get_classifier("knn", name), X_train, X_test, y_train, y_test, labels)
        else:
            return train(get_classifier("knn", name), X_train, X_test, y_train, y_test, labels)


def classify_pca(dataset, clf, name):
    data = dataset.copy()

    if name == "PD":
        data.pop("id")
        y = data.pop("class").values
        X = data.values
    else:
        data = data.groupby("Cover_Type").apply(lambda s: s.sample(100))
        y: np.ndarray = data.pop("Cover_Type").values
        X: np.ndarray = data.values

    labels: np.ndarray = pd.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)

    # normalize and balance the dataset
    X_train, X_test, y_train, y_test = norm.standard_scaler(X_train, X_test, y_train, y_test)
    X_train, y_train = balance.run(X_train, y_train, "all", 42, False)

    if clf == "Naive_Bayes":
        if name == "PD":
            return train(
                get_classifier("naive_bayes", name), X_train, X_test, y_train, y_test, labels
            )
        else:
            return train(
                get_classifier("naive_bayes", name), X_train, X_test, y_train, y_test, labels
            )

    elif clf == "KNN":
        if name == "PD":
            return train(get_classifier("knn", name), X_train, X_test, y_train, y_test, labels)
        else:
            return train(get_classifier("knn", name), X_train, X_test, y_train, y_test, labels)


#
#
#                      P E R F O R M A N C E
#
#


def performance_balance(accuracy, specificity, clf, name):
    print("SMOTE | None")
    print("\t Using Classifier:" + clf)
    print("\t\t Accuracy: ", accuracy)
    if name == "PD":
        print("\t\t Specificity: ", specificity)


def performance_normalization(accuracy, specificity, clf, name):
    print("min_max_scaler | standard_scaler | none")
    print("\t Using Classifier:" + clf)
    print("\t\t Accuracy: ", accuracy)
    if name == "PD":
        print("\t\t Specificity: ", specificity)


def performance_feature_selection(
    accuracy, specificity, features, clf, algorithm, result_features, result_perf, name
):
    max_idx_accu = np.argmax(accuracy)
    if name == "PD":
        max_idx_spec = np.argmax(specificity)

    if algorithm == "SelectKBest" or algorithm == "SelectPercentile":
        n_best_accu = features[max_idx_accu]
        if name == "PD":
            n_best_spec = features[max_idx_spec]

    accuracy = np.round(accuracy, 3)
    if name == "PD":
        specificity = np.round(specificity, 3)

    print("\t Using Classifier: " + clf)
    print("\t\t Accuracy: ", accuracy)
    if name == "PD":
        print("\t\t Specificity: ", specificity)

    if algorithm == "SelectKBest":
        print("\t\t Best accuracy: ", accuracy[max_idx_accu], "with n_features = ", n_best_accu)
        if name == "PD":
            print(
                "\t\t Best Specificity: ",
                specificity[max_idx_spec],
                "with n_features = ",
                n_best_spec,
            )

        result_features["SelectKBest"][0] += n_best_accu
        if name == "PD":
            result_features["SelectKBest"][1] += n_best_spec

        result_perf["SelectKBest"][0] += accuracy[max_idx_accu]
        if name == "PD":
            result_perf["SelectKBest"][1] += specificity[max_idx_spec]

    elif algorithm == "SelectPercentile":
        print(
            "\t\t Best accuracy: ",
            accuracy[max_idx_accu],
            "with percentage of features = ",
            n_best_accu,
            "%",
        )
        if name == "PD":
            print(
                "\t\t Best Specificity: ",
                specificity[max_idx_spec],
                "with percentage of features = ",
                n_best_spec,
                "%",
            )

        result_features["SelectPercentile"][0] += n_best_accu
        if name == "PD":
            result_features["SelectPercentile"][1] += n_best_spec

        result_perf["SelectPercentile"][0] += accuracy[max_idx_accu]
        if name == "PD":
            result_perf["SelectPercentile"][1] += specificity[max_idx_spec]

    elif algorithm == "Wrapper":
        print("\t\tBest accuracy: ", accuracy[max_idx_accu], "with n_features = ", features)
        if name == "PD":
            print(
                "\t\tBest Specificity: ", specificity[max_idx_spec], "with n_features = ", features
            )
        result_features["Wrapper"][0] += features
        if name == "PD":
            result_features["Wrapper"][1] += features

        result_perf["Wrapper"][0] += accuracy[max_idx_accu]
        if name == "PD":
            result_perf["Wrapper"][1] += specificity[max_idx_spec]


def performance_pca(accuracy, specificity, n_components, clf, name):
    max_idx_accu = np.argmax(accuracy)
    if name == "PD":
        max_idx_spec = np.argmax(specificity)

    n_best_accu = n_components[max_idx_accu]
    if name == "PD":
        n_best_spec = n_components[max_idx_spec]

    accuracy = np.round(accuracy, 3)
    if name == "PD":
        specificity = np.round(specificity, 3)

    print("\t Using Classifier: " + clf)
    print("\t\t Accuracy: ", accuracy)
    if name == "PD":
        print("\t\t Specificity: ", specificity)

    print("\t\t Best accuracy: ", accuracy[max_idx_accu], "with n_components = ", n_best_accu)
    if name == "PD":
        print(
            "\t\t Best Specificity: ",
            specificity[max_idx_spec],
            "with n_components = ",
            n_best_spec,
        )

    return


#
#
#                A U X I L I A R   F U N C T I O N S
#
#


def plot_feature_selection(xaxis, y1axis, y2axis, clf, algorithm):

    plt.subplot(2, 1, 1)
    plt.plot(xaxis, y1axis, "o-")
    plt.title("Accuracy and Specificity of " + clf)

    if algorithm == "SelectKBest":
        plt.xlabel("n_features")

    elif algorithm == "SelectPercentile":
        plt.xlabel("Percentage of n_features")

    plt.ylabel("accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(xaxis, y2axis, ".-")
    plt.xlabel("n_features")
    plt.ylabel("Specificity")

    plt.show()

    return 0


def compareFeatures(result):
    print(
        "Comparing Features: SelectKBest (n_features) | "
        "SelectPercentile (%_features) | Wrapper (n_features) |"
    )
    r = "\t\t\t"
    for feat in result.values():
        r += str(feat[0]) + " | "
    print(r)


def comparePerformance(result, name):
    print(
        "Comparing Performance [Accuracy, Specificity]: SelectKBest | SelectPercentile | Wrapper |"
    )
    r = "\t\t\t"
    for perf in result.values():
        if name == "PD":
            r += str(perf) + " | "
        else:
            r += str(perf[0]) + " | "
    print(r)


#
#
#                    P R E P R O C E S S I N G
#
#


def preprocessing(dataset, name):
    normalize_analysis(dataset, name)
    balance_analysis(dataset, name)
    feature_selection_analysis(dataset, name)
    if name == "PD":
        pca_analysis(dataset, name)
    else:
        print("PCA performance with Cover Type dataset was crashing")


# Testing
# dataset = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
# dataset = pd.read_csv('Data/covtype.csv', sep=',', decimal='.')
# preprocessing(dataset, "PD")
# normalize_analysis(dataset, "CT")
# balance_analysis(dataset, "CT")
# feature_selection_analysis(dataset, "CT")
# pca_analysis(dataset, "CT")
