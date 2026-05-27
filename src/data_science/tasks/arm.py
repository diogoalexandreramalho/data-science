"""Association Rule Mining over a dataset.

Selects features (PD only), discretizes, dummifies, then mines association rules
across multiple min_support thresholds. Plots rule-quality metrics vs support.
"""

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_selection import SelectKBest, f_classif

from data_science.datasets import DATASETS
from data_science.viz import plots as func


def run(data, source):
    target = DATASETS[source].target_column
    y = data.pop(target).values
    X = data.values

    if source == "PD":
        # Select the best 10 features that describe the model
        selector = SelectKBest(f_classif, k=10)
        kb = selector.fit_transform(X, y)
        new_data = pd.DataFrame(kb)

        # Get features selected (by name)
        selected = [col for col, keep in zip(data.columns, selector.get_support()) if keep]
        print("Features selected:")
        for feat in selected:
            print(f"  {feat}")
        print()
    else:
        new_data = data

    # Discretize
    newdf = new_data.copy()
    if source == "PD":
        for col in newdf:
            newdf[col] = pd.qcut(newdf[col], 4, labels=["0", "1", "2", "3"])
    else:
        for i, col in enumerate(newdf):
            if i < 10:
                newdf[col] = pd.qcut(newdf[col], 5, labels=["0", "1", "2", "3", "4"])
    print(newdf)

    # Dummify - one-hot encode each column
    dummified_df = pd.concat([pd.get_dummies(newdf[[att]]) for att in newdf], axis=1)
    print(dummified_df)

    # Pattern mining across min_support thresholds
    avg_confidence_list = []
    avg_lift_list = []
    avg_leverage_list = []
    number_of_rules = []
    minsup_list = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.2, 0.22]

    for sup in minsup_list:
        frequent_itemsets = apriori(dummified_df, min_support=sup, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        rules["antecedent_len"] = rules["antecedents"].apply(len)
        print(rules)

        confidence = rules["confidence"].values
        lift = rules["lift"].values
        leverage = rules["leverage"].values
        antecedent = rules["antecedents"].values
        consequent = rules["consequents"].values

        # Track top-5 rules by confidence
        top_rules_confidence = []
        top_rules = []
        for i in range(len(confidence)):
            rule = f"{tuple(antecedent[i])}->{tuple(consequent[i])}"
            if len(top_rules_confidence) < 5 and rule not in top_rules:
                top_rules_confidence.append(confidence[i])
                top_rules.append(rule)
            else:
                for x in range(len(top_rules_confidence)):
                    if confidence[i] > top_rules_confidence[x] and rule not in top_rules:
                        top_rules_confidence[x] = confidence[i]
                        top_rules[x] = rule

        avg_confidence_list.append(confidence.mean())
        avg_lift_list.append(lift.mean())
        avg_leverage_list.append(leverage.mean())
        number_of_rules.append(rules.shape[0])

        for rule in top_rules:
            print(rule)

    # Multiple Line Chart with Average measures of the rules per support
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

    # Line Chart with number of rules per support
    plt.figure(figsize=(12, 4))
    func.multiple_line_chart(
        plt.gca(),
        minsup_list,
        {"number_of_rules": number_of_rules},
        "Number of Rules per Support",
        "support",
        "",
    )
    plt.show()
