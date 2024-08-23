from xgboost import XGBRegressor
import numpy as np

class XGBoostModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1)

    def fit(self, X_train, y_train):
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_2d, y_train)

    def predict(self, X_test):
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test_2d).reshape(-1, 1)
