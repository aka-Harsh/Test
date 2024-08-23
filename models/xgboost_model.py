from xgboost import XGBRegressor

class XGBoostModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)