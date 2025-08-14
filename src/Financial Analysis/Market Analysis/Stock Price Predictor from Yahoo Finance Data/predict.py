# predict.py

import joblib
import pandas as pd
from preprocess import parse_numeric

def preprocess_input(sample_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([sample_input])
    df['change %'] = df['change %'].str.replace('%', '').str.replace('+', '').astype(float)
    df['volume'] = df['volume'].apply(parse_numeric)
    df['market_cap'] = df['market_cap'].apply(parse_numeric)
    df['pe_ratio'] = pd.to_numeric(df['pe_ratio'].replace('--', None), errors='coerce')
    df['change'] = pd.to_numeric(df['change'])
    return df

def predict_price(sample_input: dict) -> float:
    model = joblib.load("model/stock_price_predictor.pkl")
    input_df = preprocess_input(sample_input)
    return model.predict(input_df)[0]

if __name__ == "__main__":
    # 🔍 Example usage
    sample = {
        "change": "-4.15",
        "change %": "-2.33%",
        "volume": "202.637M",
        "market_cap": "4.237T",
        "pe_ratio": "56.22"
    }

    predicted_price = predict_price(sample)
    print(f"Predicted Price: ${predicted_price:.2f}")
