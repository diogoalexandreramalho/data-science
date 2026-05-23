import numpy as np

import pandas as pd
from pandas.plotting import register_matplotlib_converters

import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import plot_functions as func

def simple_knn(trnX, tstX, trnY, tstY, n, d, labels):

    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
    knn.fit(trnX, trnY)
    prdY = knn.predict(tstX)
    accuracy = metrics.accuracy_score(tstY, prdY)

    cnf_matrix = metrics.confusion_matrix(tstY, prdY, labels=labels)
    tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels=labels).ravel()
    specificity = tp/(tp+fn)

    return accuracy, specificity, cnf_matrix
    


def simple_knn_CT(trnX, tstX, trnY, tstY, n, d, labels):

    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
    knn.fit(trnX, trnY)
    prdY = knn.predict(tstX)
   
    accuracy = metrics.accuracy_score(tstY, prdY)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)
    
    return accuracy, cnf_mtx



def k_near_ngb(trnX, tstX, trnY, tstY, labels, plot):
    register_matplotlib_converters()

    
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 35, 40, 45, 50, 52, 55, 60, 70, 80, 90, 100, 105, 110, 120, 130, 140, 150]
    dist = ['manhattan', 'euclidean', 'chebyshev']

    
    acc_values = {}
    spec_values = {}
    max_accuracy = 0
    max_specificity = 0


    for d in dist:
        accuracy_values = []
        specificity_values = []

        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            accuracy = metrics.accuracy_score(tstY, prdY)
            accuracy_values.append(accuracy)

            tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels=labels).ravel()
            specificity = tp/(tp+fn)
            specificity_values.append(specificity)

            cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)
            
            if accuracy > max_accuracy:
                best_accuracy = [(d, n), accuracy, specificity, cnf_mtx]
                max_accuracy = accuracy
                
            if specificity > max_specificity:
                best_specificity = [(d, n), accuracy, specificity, cnf_mtx]
                max_specificity = specificity
        
        acc_values[d] = accuracy_values
        spec_values[d] = specificity_values



    if plot:
        plt.figure()
        func.multiple_line_chart(plt.gca(), nvalues, acc_values, 'KNN variants', 'n', 'accuracy', percentage=True)
        plt.show()
        func.multiple_line_chart(plt.gca(), nvalues, spec_values, 'KNN variants', 'n', 'specificity', percentage=True)
        plt.show()

    return ["kNN", best_accuracy, best_specificity]



def k_near_ngb_CT(trnX, tstX, trnY, tstY, labels, plot):
    register_matplotlib_converters()


    nvalues = [1, 3, 10, 15, 17, 19, 31, 35, 50, 52, 55, 70, 90, 100, 120, 130, 150]
    dist = ['manhattan', 'euclidean', 'chebyshev']

    acc_values = {}
    max_accuracy = 0


    for d in dist:
        accuracy_values = []

        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            accuracy = metrics.accuracy_score(tstY, prdY)
            accuracy_values.append(accuracy)

            cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)
            
            if accuracy > max_accuracy:
                best_accuracy = [(d, n), accuracy, cnf_mtx]
                max_accuracy = accuracy
                
        acc_values[d] = accuracy_values


    if plot:
        plt.figure()
        func.multiple_line_chart(plt.gca(), nvalues, acc_values, 'KNN variants', 'n', 'accuracy', percentage=True)
        plt.show()

    return ["kNN", best_accuracy]