# predict.py
import pandas as pd
import joblib
import os

MODEL_PATH = "model/roi_predictor.pkl"

def predict_sales(sample_input: dict) -> float:
    """
    Predict product sales based on input features.
    Generates start_month and start_dayofweek if start_date is given.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Train the model first.")

    # Handle date conversion if raw date provided
    if "start_date" in sample_input:
        date_val = pd.to_datetime(sample_input["start_date"], errors="coerce")
        if pd.isna(date_val):
            raise ValueError(f"Invalid start_date: {sample_input['start_date']}")
        sample_input["start_month"] = date_val.month
        sample_input["start_dayofweek"] = date_val.dayofweek
        del sample_input["start_date"]  # not needed for prediction

    # Check that required columns are present
    required_cols = [
        "platform",
        "influencer_category",
        "campaign_type",
        "engagements",
        "estimated_reach",
        "campaign_duration_days",
        "start_month",
        "start_dayofweek"
    ]
    missing = set(required_cols) - set(sample_input.keys())
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Load trained pipeline
    pipeline = joblib.load(MODEL_PATH)

    # Create DataFrame and predict
    df = pd.DataFrame([sample_input])
    prediction = pipeline.predict(df)[0]
    return float(prediction)

if __name__ == "__main__":
    # You can give either start_date OR start_month & start_dayofweek
    sample = {
        "platform": "YouTube",
        "influencer_category": "Food",
        "campaign_type": "Product Launch",
        "engagements": 50000,
        "estimated_reach": 300000,
        "campaign_duration_days": 14,
        "start_date": "2024-01-09"  # Will be split automatically
    }

    pred = predict_sales(sample)
    print(f"Predicted Product Sales: {pred:.2f}")
