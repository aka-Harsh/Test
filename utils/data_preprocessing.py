import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    X = data[features].values
    y = data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    return X_scaled, y_scaled, scaler

def create_sequences(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps])
    
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
