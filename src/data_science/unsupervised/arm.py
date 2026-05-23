from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules  # for ARM
from sklearn.feature_selection import SelectKBest, f_classif

from data_science.viz import plots as func

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def run(source, data, group=None):

    dataPD = pd.read_csv(
        DATA_DIR / "raw" / "pd_speech_features.csv", sep=",", decimal=".", skiprows=1
    )
    dataCT = pd.read_csv(DATA_DIR / "raw" / "covtype_test.data", sep=",", decimal=".")

    if source == "PD":
        target = "class"
        data = dataPD
    else:
        target = "Cover_Type"
        data = dataCT

    y = data.pop(target).values
    X: np.ndarray = data.values
    # labels = pd.unique(y)

    if source == "PD":
        # Select the best 10 features that describe the model
        selector = SelectKBest(f_classif, k=10)
        kb = selector.fit_transform(X, y)

        new_data = pd.DataFrame(kb)

        # Get Features Selected Names
        features_selected_bool = selector.get_support()
        i = 0
        features_selected_list = []
        for col in data.columns:
            if features_selected_bool[i]:
                features_selected_list.append(col)
            i += 1

        print("Features Selected : \n")
        for feat in features_selected_list:
            print(feat)
        print("\n")

    else:
        new_data = data

    # Discretize  - Discretization for dataset1
    if source == "PD":
        newdf = new_data.copy()
        for col in newdf:
            newdf[col] = pd.qcut(newdf[col], 4, labels=["0", "1", "2", "3"])
        print(newdf)
    else:
        # Discretize  - Discretization for dataset2
        i = 0
        newdf = new_data.copy()
        for col in newdf:
            if i < 10:
                newdf[col] = pd.qcut(newdf[col], 5, labels=["0", "1", "2", "3", "4"])
            i += 1
        print(newdf)

    # Dummify - for each nominal variable, create additional variables for each possible value
    dummylist = []
    for att in newdf:
        dummylist.append(pd.get_dummies(newdf[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)

    print(dummified_df)

    # if source == "CT":
    #    dummified_df = dummified_df.drop(columns=["y0"])

    ######## Pattern Mining #############

    avg_confidence_list = []
    avg_lift_list = []
    avg_leverage_list = []
    number_of_rules = []
    minsup_list = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.2, 0.22]
    for sup in minsup_list:
        frequent_itemsets = apriori(dummified_df, min_support=sup, use_colnames=True)
        minconf = 0.6
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
        print("passei")
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
        print(rules)

        confidence = rules["confidence"].values
        lift = rules["lift"].values
        leverage = rules["leverage"].values
        antecedent = rules["antecedents"].values
        consequent = rules["consequents"].values

        avg_confidence = 0
        avg_lift = 0
        avg_leverage = 0

        top_rules_size = 5

        top_rules_confidence = []
        top_rules = []

        for i in range(len(confidence)):
            avg_confidence += confidence[i]
            avg_lift += lift[i]
            avg_leverage += leverage[i]

            rule = str(tuple(antecedent[i])) + "->" + str(tuple(consequent[i]))
            if len(top_rules_confidence) < top_rules_size and rule not in top_rules:
                top_rules_confidence.append(confidence[i])
                top_rules.append(rule)
            else:
                for x in range(len(top_rules_confidence)):
                    if confidence[i] > top_rules_confidence[x] and rule not in top_rules:
                        top_rules_confidence[x] = confidence[i]
                        top_rules[x] = rule

        avg_confidence /= len(confidence)
        avg_lift /= len(lift)
        avg_leverage /= len(leverage)
        # avg_top_rules_confidence = sum(top_rules_confidence) / len(top_rules_confidence)

        avg_confidence_list.append(avg_confidence)
        avg_lift_list.append(avg_lift)
        avg_leverage_list.append(avg_leverage)
        number_of_rules.append(rules.shape[0])
        # avg_top_rules_list.append(avg_top_rules_confidence)

        for rule in top_rules:
            print(rule)

    ###### Multiple Line Chart with Average measures of the rules per support ######

    plt.figure(figsize=(12, 4))
    series = {
        "leverage": avg_leverage_list,
        "avg_confidence": avg_confidence_list,
        "avg_lift": avg_lift_list,
    }
    func.multiple_line_chart(
        plt.gca(), minsup_list, series, "Rules Quality per Support", "support", ""
    )
    plt.show()

    ###### Line Chart with number of rules per support  ######

    plt.figure(figsize=(12, 4))
    series = {"number_of_rules": number_of_rules}
    func.multiple_line_chart(
        plt.gca(), minsup_list, series, "Number of Rules per Support", "support", ""
    )
    plt.show()


def statistics(source, data):

    print("1. Pattern Mining")
    print(" List of supports used : [0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8]")
    print(" Rules Quality measures used : confidence, lift, leverage")
