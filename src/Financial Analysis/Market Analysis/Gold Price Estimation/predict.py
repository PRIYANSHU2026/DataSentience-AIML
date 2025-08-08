import pandas as pd
import joblib
from preprocess import preprocess_data

def predict_price(sample_input: dict) -> float:
    """
    Predicts gold priceClose given input features.

    Args:
        sample_input (dict): Dictionary with keys: priceOpen, priceHigh, priceLow, volume

    Returns:
        float: Predicted priceClose value
    """
    model = joblib.load("model/gold_price_model.pkl")
    df = pd.DataFrame([sample_input])
    return model.predict(df)[0]

if __name__ == "__main__":
    # 🔍 Sample Input
    sample_input = {
        "priceOpen": 0.015,
        "priceHigh": 0.016,
        "priceLow": 0.014,
        "volume": 350.0
    }

    # 🔮 Make prediction
    predicted_price = predict_price(sample_input)

    # 🖨️ Display Result
    print("📈 Predicted priceClose:", round(predicted_price, 6))
