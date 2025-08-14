import joblib
import numpy as np

def predict_risk(sample_input):
    """
    sample_input = [Exchange_Vol, Equity_Vol, Yield_Change, Bond_Yield, Oil_Price]
    """
    model = joblib.load("models/risk_classifier.pkl")
    prediction = model.predict([sample_input])[0]
    print(f"🔍 Predicted Country Risk Level: {prediction}")

if __name__ == "__main__":
    # Example: [0.012, 0.008, 0.15, 7.9, 70.22]
    sample = [0.012, 0.008, 0.15, 7.9, 70.22]
    predict_risk(sample)
