import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.lstm_model import LSTMModel
from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel
from utils.data_preprocessing import preprocess_data, create_sequences
from utils.evaluation import evaluate_model

def plot_predictions(data, predictions, days):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(predictions):], data['Close'][-len(predictions):], label='Actual')
    for model, pred in predictions.items():
        plt.plot(data.index[-len(predictions):], pred, label=f'{model} Prediction')
    plt.plot(pd.date_range(start=data.index[-1], periods=days+1, freq='D')[1:], 
             predictions['LSTM'][-days:], '--', label='Future Prediction')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    return plt

def main():
    st.title("Stock Market Price Predictor")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
        st.write("Data preview:")
        st.write(data.head())

        days_to_predict = st.slider("Select number of days to predict", 1, 100, 30)

        if st.button("Predict Future Values"):
            X, y, scaler = preprocess_data(data)
            X_train, X_test, y_train, y_test = create_sequences(X, y)

            models = {
                "LSTM": LSTMModel(),
                "XGBoost": XGBoostModel(),
                "Random Forest": RandomForestModel()
            }

            predictions = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = evaluate_model(y_test, y_pred)
                st.write(f"{name} Model Prediction Accuracy: {accuracy:.2f}%")
                
                # Predict future values
                last_sequence = X_test[-1].reshape(1, -1, 5)
                future_pred = []
                for _ in range(days_to_predict):
                    next_pred = model.predict(last_sequence)
                    future_pred.append(next_pred[0])
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1, 0] = next_pred[0]
                
                predictions[name] = scaler.inverse_transform(np.concatenate([y_pred, future_pred]))[:, 0]

            # Plot results
            fig = plot_predictions(data, predictions, days_to_predict)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
