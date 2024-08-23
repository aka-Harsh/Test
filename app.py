import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.lstm_model import LSTMModel
from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel
from utils.data_preprocessing import preprocess_data, create_sequences
from utils.evaluation import calculate_probabilities

def plot_predictions(data, predictions, days, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-60:], data['Close'][-60:], label='Historical', color='blue')
    
    future_dates = pd.date_range(start=data.index[-1], periods=days+1, freq='D')[1:]
    plt.plot(future_dates, predictions, label=f'{model_name} Prediction', color='red')
    
    plt.title(f'{model_name} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Add arrow annotations
    last_historical = data['Close'].iloc[-1]
    first_prediction = predictions[0]
    mid_prediction = predictions[len(predictions)//2]
    last_prediction = predictions[-1]
    
    plt.annotate('', xy=(future_dates[0], first_prediction), xytext=(data.index[-1], last_historical),
                 arrowprops=dict(facecolor='green' if first_prediction > last_historical else 'red', shrink=0.05))
    plt.annotate('', xy=(future_dates[-1], last_prediction), xytext=(future_dates[len(future_dates)//2], mid_prediction),
                 arrowprops=dict(facecolor='green' if last_prediction > mid_prediction else 'red', shrink=0.05))
    
    return plt

def get_investment_recommendation(rise_prob):
    if rise_prob >= 60:
        return "Invest"
    elif rise_prob <= 40:
        return "Don't Invest"
    else:
        return "Hold"

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

            results = []

            for name, model in models.items():
                model.fit(X_train, y_train)
                
                # Predict future values
                last_sequence = X_test[-1].reshape(1, -1, 5)
                future_pred = []
                for _ in range(days_to_predict):
                    next_pred = model.predict(last_sequence)
                    future_pred.append(next_pred[0])
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1, 0] = next_pred[0]
                
                predictions = scaler.inverse_transform(np.array(future_pred)).flatten()

                # Plot results
                fig = plot_predictions(data, predictions, days_to_predict, name)
                st.pyplot(fig)
                plt.close()

                # Calculate probabilities and price change
                rise_prob, fall_prob = calculate_probabilities(predictions)
                price_change = predictions[-1] - data['Close'].iloc[-1]
                price_change_percent = (price_change / data['Close'].iloc[-1]) * 100

                # Store results
                results.append({
                    "Model": name,
                    "Rise Probability": f"{rise_prob:.2f}%",
                    "Fall Probability": f"{fall_prob:.2f}%",
                    "Price Change": f"${price_change:.2f} ({price_change_percent:.2f}%)",
                    "Recommendation": get_investment_recommendation(rise_prob)
                })

            # Create and display the matrix table
            results_df = pd.DataFrame(results)
            st.write("Model Comparison Matrix:")
            st.table(results_df.set_index('Model'))

if __name__ == "__main__":
    main()
