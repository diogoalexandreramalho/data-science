import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

import data_balancing as balance
import Normalize as norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import KNN as knn
import Decision_Tree as dt
import naive_bayes as nb
import RandomForest as rf
import GradientBoost as gb
import XGBoost as xgb

import print_statistics as stats


def classification(data, source):

    if source == "CT":
        target = 'Cover_Type'
    else:
        target = 'class'

    # split data set in target variable and atributes
    data = data.apply(pd.to_numeric)
    y: np.ndarray = data.pop(target).values.astype(int)
    X: np.ndarray = data.values.astype(float)
    labels: np.ndarray = pd.unique(y)

    # store accuracies and sensitivities for each classifier
    accuracys = {"nb":[], "knn":[], "dt":[], "rf":[], "gb":[], "xgb":[]}
    if source == "PD":
        specificities = {"nb":[], "knn":[], "dt":[], "rf":[], "gb":[], "xgb":[]}
    
    if source == "PD":
        c_m = np.zeros((2,2), dtype=int)
        cnf_mtxs_temp = [c_m, c_m, c_m, c_m, c_m, c_m]
        cnf_mtxs = [c_m, c_m, c_m, c_m, c_m, c_m]
    else:
        c_m = np.zeros((7,7), dtype=int)
        cnf_mtxs_temp = [c_m, c_m, c_m, c_m, c_m, c_m]
        cnf_mtxs = [c_m, c_m, c_m, c_m, c_m, c_m]

    cv = KFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in cv.split(X):
        
        trnX, tstX, trnY, tstY = X[train_index], X[test_index], y[train_index], y[test_index]
        
        # normalize and balance the dataset
        trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
        trnX, trnY = balance.run(trnX, trnY, 'all', 42, False)
        
        # classify with fixed parameters and get the metrics
        acc = [0,0,0,0,0,0]

        if source == "PD":
            spec = [0,0,0,0,0,0]

            acc[0], spec[0], cnf_mtxs_temp[0] = nb.simple_naive_bayes(trnX, tstX, trnY, tstY, labels)
            acc[1], spec[1], cnf_mtxs_temp[1] = knn.simple_knn(trnX, tstX, trnY, tstY, 1, 'manhattan', labels)
            acc[2], spec[2], cnf_mtxs_temp[2] = dt.simple_decision_tree(trnX, tstX, trnY, tstY, 0.05, 5, 'entropy', labels)
            acc[3], spec[3], cnf_mtxs_temp[3] = rf.simple_random_forest(trnX, tstX, trnY, tstY, 150, 10, 'sqrt', labels)
            acc[4], spec[4], cnf_mtxs_temp[4] = gb.simple_gradient_boost(trnX, tstX, trnY, tstY, 100, 0.1, 5, 'sqrt', labels)
            acc[5], spec[5], cnf_mtxs_temp[5] = xgb.simple_xg_boost(trnX, tstX, trnY, tstY, 200, 5, labels)
        else:
            acc[0], cnf_mtxs_temp[0] = nb.simple_naive_bayes_CT(trnX, tstX, trnY, tstY, labels)
            acc[1], cnf_mtxs_temp[1] = knn.simple_knn_CT(trnX, tstX, trnY, tstY, 1, 'manhattan', labels)
            acc[2], cnf_mtxs_temp[2] = dt.simple_decision_tree_CT(trnX, tstX, trnY, tstY, 0.05, 5, 'entropy', labels)
            acc[3], cnf_mtxs_temp[3] = rf.simple_random_forest_CT(trnX, tstX, trnY, tstY, 150, 10, 'sqrt', labels)
            acc[4], cnf_mtxs_temp[4] = gb.simple_gradient_boost_CT(trnX, tstX, trnY, tstY, 100, 0.1, 5, 'sqrt', labels)
            acc[5], cnf_mtxs_temp[5] = xgb.simple_xg_boost_CT(trnX, tstX, trnY, tstY, 200, 5, labels)

        
        # add confusion matrixes
        for i in range(6):
            cnf_mtxs[i] = np.add(cnf_mtxs[i], cnf_mtxs_temp[i])

        # store metrics
        i = 0
        for clf in accuracys:
            accuracys[clf] += [acc[i]]
            if source == "PD":
                specificities[clf] += [spec[i]]
            i += 1
        print("1")
    

    # calculate avg accuracy and avg specificity 
    avg_accuracys = []
    avg_specificities = []
    for clf in accuracys:
        avg_acc_clf = sum(accuracys[clf])/len(accuracys[clf])
        avg_accuracys += [avg_acc_clf]
        if source == "PD":
            avg_spec_clf = sum(specificities[clf])/len(specificities[clf])
            avg_specificities += [avg_spec_clf]


    clf_names = ["Naive Bayes", "kNN", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]

    # create report for each classifier
    if source == "PD":
        params = [('GaussianNB'), ('manhattan', 1), ('entropy', 5, 0.05), ('sqrt', 10, 150), ('sqrt', 5, 100, 0.1), (5, 200)]

        nb_report = [ clf_names[0], params[0], avg_accuracys[0], avg_specificities[0], cnf_mtxs[0]]
        knn_report =[ clf_names[1], params[1], avg_accuracys[1], avg_specificities[1], cnf_mtxs[1]]
        dt_report = [ clf_names[2], params[2], avg_accuracys[2], avg_specificities[2], cnf_mtxs[2]]
        rf_report = [ clf_names[3], params[3], avg_accuracys[3], avg_specificities[3], cnf_mtxs[3]]
        gb_report = [ clf_names[4], params[4], avg_accuracys[4], avg_specificities[4], cnf_mtxs[4]]
        xgb_report =[ clf_names[5], params[5], avg_accuracys[5], avg_specificities[5], cnf_mtxs[5]]

        reports = [nb_report, knn_report, dt_report, rf_report, gb_report, xgb_report]

        stats.print_report(reports, (True, True))
    else:
        params = [('GaussianNB'), ('manhattan', 1), ('entropy', 50, 0.00005), ('sqrt', 25, 185), ('sqrt', 10, 300, 0.05), (10, 300)]
        
        nb_report = [clf_names[0], [params[0], avg_accuracys[0], cnf_mtxs[0]]]
        knn_report =[clf_names[1], [params[1], avg_accuracys[1], cnf_mtxs[1]]]
        dt_report = [clf_names[2], [params[2], avg_accuracys[2], cnf_mtxs[2]]]
        rf_report = [clf_names[3], [params[3], avg_accuracys[3], cnf_mtxs[3]]]
        gb_report = [clf_names[4], [params[4], avg_accuracys[4], cnf_mtxs[4]]]
        xgb_report =[clf_names[5], [params[5], avg_accuracys[5], cnf_mtxs[5]]]

        reports = [nb_report, knn_report, dt_report, rf_report, gb_report, xgb_report]

        stats.print_analysis_CT(reports, (True, True))




