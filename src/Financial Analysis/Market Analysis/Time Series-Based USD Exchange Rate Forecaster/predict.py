import pandas as pd
import joblib
from prophet import Prophet
from preprocess import load_and_preprocess
from datetime import timedelta

def predict_next_day():
    df = load_and_preprocess("data/exchange_data.csv")
    model = joblib.load("models/prophet_model.pkl")

    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)

    next_day = forecast.iloc[-1][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    print("📅 Prediction for:", next_day['ds'].date())
    print(f"💵 USD Exchange Rate (Predicted): {next_day['yhat']:.4f}")
    print(f"📉 Lower bound: {next_day['yhat_lower']:.4f}")
    print(f"📈 Upper bound: {next_day['yhat_upper']:.4f}")

if __name__ == "__main__":
    predict_next_day()
