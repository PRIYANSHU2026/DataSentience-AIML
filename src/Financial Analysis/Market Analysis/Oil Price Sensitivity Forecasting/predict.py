import joblib
import numpy as np

def predict_oil_price(new_input):
    """
    new_input = [ExchangeRate, PolicyRate, BondYield, YieldSpread, EquityIndex]
    """
    model = joblib.load("models/oil_price_model.pkl")
    input_array = np.array(new_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    print(f"⛽ Predicted Oil Price: ${prediction:.2f} per barrel")

if __name__ == "__main__":
    # Sample input
    sample = [5.00, 5.01, 7.10, 4.10, 1002.29]
    predict_oil_price(sample)
