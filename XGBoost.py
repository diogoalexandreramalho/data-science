import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import sklearn.metrics as metrics
import plot_functions as func


def simple_xg_boost(trnX, tstX, trnY, tstY, n, d, labels):

    xgb = XGBClassifier(n_estimators=n, max_depth=d)
    xgb.fit(trnX, trnY)
    prdY = xgb.predict(tstX)
    accuracy = metrics.accuracy_score(tstY, prdY)

    cnf_matrix = metrics.confusion_matrix(tstY, prdY, labels=labels)
    tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels=labels).ravel()
    specificity = tp/(tp+fn)

    return accuracy, specificity, cnf_matrix


def simple_xg_boost_CT(trnX, tstX, trnY, tstY, n, d, labels):

    xgb = XGBClassifier(n_estimators=n, max_depth=d)
    xgb.fit(trnX, trnY)
    prdY = xgb.predict(tstX)
    
    accuracy = metrics.accuracy_score(tstY, prdY)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)

    return accuracy, cnf_mtx


def xg_boost_CT(trnX, tstX, trnY, tstY, labels, plot):
    

    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]

    max_accuracy = 0
    acc_values = {}
    
    plt.figure()
    
    
    for d in max_depths:
        accuracy_values = []
        for n in n_estimators:
            xgb = XGBClassifier(n_estimators=n, max_depth=d)
            xgb.fit(trnX, trnY)
            prdY = xgb.predict(tstX)

            accuracy = metrics.accuracy_score(tstY, prdY)
            accuracy_values.append(accuracy)

            cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels=labels)

            if accuracy > max_accuracy:
                best_accuracy = [(d, n), accuracy, cnf_mtx]
                max_accuracy = accuracy
                

        acc_values[d] = accuracy_values

                
        func.multiple_line_chart(plt.gca(), n_estimators, acc_values, 'XG Boost', 'nr estimators', 
                                 'accuracy', percentage=True)

    if plot:
        plt.show()
        

    return ["XGBoost", best_accuracy]


def xg_boost(trnX, tstX, trnY, tstY, labels, plot):
    

    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]

    max_accuracy = 0
    max_specificity = 0


    plt.figure()
    
    acc_values = {}
    spec_values = {}
    
    for d in max_depths:
        accuracy_values = []
        specificity_values = []
        for n in n_estimators:
            xgb = XGBClassifier(n_estimators=n, max_depth=d)
            xgb.fit(trnX, trnY)
            prdY = xgb.predict(tstX)

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

                
        """func.multiple_line_chart(plt.gca(), n_estimators, acc_values, 'XG Boost', 'nr estimators', 
                                 'accuracy', percentage=True)"""
        func.multiple_line_chart(plt.gca(), n_estimators, spec_values, 'XG Boost', 'nr estimators', 
                                 'specificity', percentage=True)

    if plot:
        plt.show()
        

    return ["XGBoost", best_accuracy, best_specificity]
    

