import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import plot_functions as func


def simple_random_forest(trnX, tstX, trnY, tstY, n, d, f, labels):

    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
    rf.fit(trnX, trnY)
    prdY = rf.predict(tstX)
    accuracy = metrics.accuracy_score(tstY, prdY)

    cnf_matrix = metrics.confusion_matrix(tstY, prdY, labels=labels)
    tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels=labels).ravel()
    specificity = tp/(tp+fn)

    return accuracy, specificity, cnf_matrix



def simple_random_forest_CT(trnX, tstX, trnY, tstY, n, d, f, labels):

    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
    rf.fit(trnX, trnY)
    prdY = rf.predict(tstX)
    
    accuracy = metrics.accuracy_score(tstY, prdY)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)

    return accuracy, cnf_mtx



def random_forest_CT(trnX, tstX, trnY, tstY, labels, plot):

    n_estimators = [5, 10, 25, 50, 75, 100, 110,130,150,170,185, 200, 250, 300, 350,400, 450, 500]
    max_depths = [5, 10, 25, 30, 40, 50, 60, 70]
    max_features = ['sqrt', 'log2']

    max_accuracy = 0

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        acc_values = {}
        for d in max_depths:
            accuracy_values = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)

                # accuracy for max_features = f, max_depth = d, n_estimators = n
                accuracy = metrics.accuracy_score(tstY, prdY)
                accuracy_values.append(accuracy)

                cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)

                if accuracy > max_accuracy:
                    best_accuracy = [(f, d, n), accuracy, cnf_mtx]
                    max_accuracy = accuracy
                

            acc_values[d] = accuracy_values

        func.multiple_line_chart(axs[0, k], n_estimators, acc_values, 'Random Forests with %s features'%f, 'nr estimators', 
                                 'accuracy', percentage=True)

    if plot:
        plt.show()
    
    
    return ["Random Forest", best_accuracy]



def random_forest(trnX, tstX, trnY, tstY, labels, plot):

    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    max_depths = [5, 10, 25, 40, 50, 60, 70]
    max_features = ['sqrt', 'log2']

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
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)

                # accuracy for max_features = f, max_depth = d, n_estimators = n
                accuracy = metrics.accuracy_score(tstY, prdY)
                accuracy_values.append(accuracy)

                # sensitivity for max_features = f, max_depth = d, n_estimators = n
                tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels=labels).ravel()
                specificity = tp/(tp+fn)
                specificity_values.append(specificity)

                cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)

                if accuracy > max_accuracy:
                    best_accuracy = [(f, d, n), accuracy, specificity, cnf_mtx]
                    max_accuracy = accuracy
                
                if specificity > max_specificity:
                    best_specificity = [(f, d, n), accuracy, specificity, cnf_mtx]
                    max_specificity = specificity

            acc_values[d] = accuracy_values
            spec_values[d] = specificity_values

        """func.multiple_line_chart(axs[0, k], n_estimators, acc_values, 'Random Forests with %s features'%f, 'nr estimators', 
                                 'accuracy', percentage=True)"""
        func.multiple_line_chart(axs[0, k], n_estimators, spec_values, 'Random Forests with %s features'%f, 'nr estimators', 
                                 'specificity', percentage=True)

    if plot:
        plt.show()
    
    
    return ["Random Forest", best_accuracy, best_specificity]

