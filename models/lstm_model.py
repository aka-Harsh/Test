import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def __init__(self):
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, 5)),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def fit(self, X_train, y_train):
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    def predict(self, X_test):
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        return self.model.predict(X_test).flatten()