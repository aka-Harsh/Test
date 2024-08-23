import streamlit as st
import pandas as pd
from models.lstm_model import LSTMModel
from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel
from utils.data_preprocessing import preprocess_data
from utils.evaluation import evaluate_model

def main():
    st.title("Stock Market Price Predictor")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(data.head())

        if st.button("Predict Future Values"):
            X_train, X_test, y_train, y_test = preprocess_data(data)

            models = [
                ("LSTM", LSTMModel()),
                ("XGBoost", XGBoostModel()),
                ("Random Forest", RandomForestModel())
            ]

            for name, model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = evaluate_model(y_test, y_pred)
                st.write(f"{name} Model Prediction Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
