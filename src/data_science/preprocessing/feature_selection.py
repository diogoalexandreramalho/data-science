import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif


# Note: Feature Selection removes ID for PD dataset
def feature_selection(dataset, name):
    data = dataset.copy()
    # Prepare data
    if name == "PD":  # Pd_dataset
        data["class"]  # Save the class for later
        data.pop("id")  # Remove id, because we don't care, maybe remove at the begining
        y = data.pop("class").values
        X = data.values

        features_list = data.columns.copy()
        selector = SelectPercentile(f_classif, percentile=55)
        X_new = selector.fit_transform(X, y)

        mask = selector.get_support()
        X_new_data = pd.DataFrame(X_new)

        # This gets me the an array with the names of the features that were selected
        new_features = []  # The list of your K best features
        for bool, feature in zip(mask, features_list):
            if bool:
                new_features.append(feature)

        # create new dataset
        new_data = X_new_data.set_axis(new_features, axis=1, inplace=False)
        new_data["class"] = y

        return new_data

    else:  # Cover_Type
        data["Cover_Type"]
        y = data.pop("Cover_Type")
        X = data.values

        features_list = data.columns.copy()

        return


# Testing

# dataset = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
# dataset = pd.read_csv('covtype.data', sep=',', decimal='.')
# data = dataset.copy()
# feature_selection_analysis(data, True)
# new_data = feature_selection(data, True).copy()
# print(new_data)
