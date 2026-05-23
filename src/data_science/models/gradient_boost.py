import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier

from data_science.viz import plots as func


def simple_gradient_boost(trnX, tstX, trnY, tstY, n, lr, d, f, labels):

    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=d, max_features=f)
    gb.fit(trnX, trnY)
    prdY = gb.predict(tstX)
    accuracy = metrics.accuracy_score(tstY, prdY)

    cnf_matrix = metrics.confusion_matrix(tstY, prdY, labels=labels)
    tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels=labels).ravel()
    specificity = tp / (tp + fn)

    return accuracy, specificity, cnf_matrix


def simple_gradient_boost_CT(trnX, tstX, trnY, tstY, n, lr, d, f, labels):

    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=d, max_features=f)
    gb.fit(trnX, trnY)
    prdY = gb.predict(tstX)

    accuracy = metrics.accuracy_score(tstY, prdY)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)

    return accuracy, cnf_mtx


def gradient_boost_CT(trnX, tstX, trnY, tstY, labels, plot):

    lr_list = [0.05, 0.075, 0.1, 0.5, 1]
    n_estimators = [5, 20, 50, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]
    max_features = ["sqrt", "log2"]

    max_accuracy = 0

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        acc_values = {}
        for d in max_depths:
            accuracy_values = []
            for n in n_estimators:
                max_learning_accuracy = 0

                for lr in lr_list:
                    gb = GradientBoostingClassifier(
                        n_estimators=n, learning_rate=lr, max_depth=d, max_features=f
                    )
                    gb.fit(trnX, trnY)
                    prdY = gb.predict(tstX)

                    accuracy = metrics.accuracy_score(tstY, prdY)

                    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)

                    if accuracy > max_accuracy:
                        best_accuracy = [(f, d, n, lr), accuracy, cnf_mtx]
                        max_accuracy = accuracy

                    if accuracy > max_learning_accuracy:
                        max_learning_accuracy = accuracy

                accuracy_values.append(max_learning_accuracy)

            acc_values[d] = accuracy_values

        func.multiple_line_chart(
            axs[0, k],
            n_estimators,
            acc_values,
            f"Gradient Boost with {f} features",
            "nr estimators",
            "accuracy",
            percentage=True,
        )

    if plot:
        plt.show()

    return ["Gradient Boost", best_accuracy]


def gradient_boost(trnX, tstX, trnY, tstY, labels, plot):

    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]
    max_features = ["sqrt", "log2"]

    max_accuracy = 0
    max_specificity = 0

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        acc_values = {}
        spec_values = {}
        for d in max_depths:
            accuracy_values = []
            specificity_values = []
            for n in n_estimators:
                max_learning_accuracy = 0
                max_learning_specificity = 0

                for lr in lr_list:
                    gb = GradientBoostingClassifier(
                        n_estimators=n, learning_rate=lr, max_depth=d, max_features=f
                    )
                    gb.fit(trnX, trnY)
                    prdY = gb.predict(tstX)

                    accuracy = metrics.accuracy_score(tstY, prdY)

                    tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels=labels).ravel()
                    specificity = tp / (tp + fn)

                    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)

                    if accuracy > max_accuracy:
                        best_accuracy = [(f, d, n, lr), accuracy, specificity, cnf_mtx]
                        max_accuracy = accuracy

                    if specificity > max_specificity:
                        best_specificity = [(f, d, n, lr), accuracy, specificity, cnf_mtx]
                        max_specificity = specificity

                    if accuracy > max_learning_accuracy:
                        max_learning_accuracy = accuracy

                    if specificity > max_learning_specificity:
                        max_learning_specificity = specificity

                accuracy_values.append(max_learning_accuracy)
                specificity_values.append(max_learning_specificity)

            acc_values[d] = accuracy_values
            spec_values[d] = specificity_values

        """func.multiple_line_chart(axs[0, k], n_estimators, acc_values,
                                 'Gradient Boost with %s features'%f, 'nr estimators',
                                 'accuracy', percentage=True)"""
        func.multiple_line_chart(
            axs[0, k],
            n_estimators,
            spec_values,
            f"Gradient Boost with {f} features",
            "nr estimators",
            "specificity",
            percentage=True,
        )

    if plot:
        plt.show()

    return ["Gradient Boost", best_accuracy, best_specificity]
