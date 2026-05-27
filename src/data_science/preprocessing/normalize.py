from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standard_scaler(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def min_max_scaler(X_train, X_test, y_train, y_test):

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
