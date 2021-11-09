from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def goal_function(clf, X, y, seed=1410, times_cross_validation=3):
        scaler = MinMaxScaler()
        kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=seed)
        result = np.zeros(times_cross_validation)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            result[i] =  mean_absolute_error(y_test, y_pred)
        return round(result.mean(), 8)