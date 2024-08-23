from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)