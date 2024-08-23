import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def __init__(self):
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(60, 5), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    def predict(self, X_test):
        return self.model.predict(X_test)
