import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    accuracy = 100 - (rmse / np.mean(y_true) * 100)
    return accuracy