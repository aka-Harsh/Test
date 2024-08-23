import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    # Assuming the CSV file has columns: Date, Open, High, Low, Close, Volume
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    target = 'Close'

    X = data[features]
    y = data[target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test