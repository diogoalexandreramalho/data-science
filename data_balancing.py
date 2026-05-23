import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import plot_functions as func

from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler

import Normalize as norm
import naive_bayes as nb
import KNN as knn
import Decision_Tree as dt

'''
Prints the values for the class, and plots a bar_chart showing the
    difference
'''
def unbalanced_data(data):
    target_count = data['class'].value_counts()
    """
    plt.figure()
    plt.title('Class balance')
    plt.bar(target_count.index, target_count.values)"""

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print('Minority class(', ind_min_class, '):', target_count[ind_min_class]) 
    print('Majority class(', 1-ind_min_class, '):',  target_count[1 - ind_min_class]) 
    print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')

    #plt.show() 
    return 0


# balance by oversampling the minority class using SMOTE
def balance_SMOTE(data, strategy: str, plot, random_state=42):
    target_count = data['class'].value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}

    df_class_min = data[data['class'] == min_class]
    df_class_max = data[data['class'] != min_class]

    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1 - ind_min_class]]

    smote = SMOTE(sampling_strategy=strategy, random_state=random_state)
    y = data.pop('class').values
    X = data.values
    smote_X, smote_y = smote.fit_resample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1 - ind_min_class]]
    
    if plot:
        plt.figure()
        func.multiple_bar_chart(plt.gca(), 
                            [target_count.index[ind_min_class], target_count.index[1-ind_min_class]], 
                            values, 'Target', 'frequency', 'Class balance')

        plt.show()

    df_smote_y = pd.DataFrame(smote_y, columns=['class'])
    new_data = pd.concat([pd.DataFrame(smote_X), df_smote_y], axis=1)

    return new_data    


def run(trnX, trnY, strategy: str, random_number, plot):
    register_matplotlib_converters()
    
    # create dataframe from training set
    df_trnX = pd.DataFrame(trnX)
    df_trnY = pd.DataFrame(trnY, columns = ['class'])
    df_trn = pd.concat([df_trnX, df_trnY], axis=1, sort=False)

    if plot:
        unbalanced_data(df_trn) #Shows unbalnced data

    # balance training set
    df_trn = balance_SMOTE(df_trn, strategy, plot, random_number) # Shows Smote, and returns new_data

    # split training set in attributes and target
    trnY = df_trn.pop('class').values
    trnX = df_trn.values
    
    return trnX, trnY