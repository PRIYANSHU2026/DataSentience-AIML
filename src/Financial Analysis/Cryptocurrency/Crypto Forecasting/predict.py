import pandas as pd
import argparse
import os
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to predict for (e.g., BTCUSDT)")
    parser.add_argument("--days", type=int, default=1, help="Number of future days to predict")
    args = parser.parse_args()

    processed_path = os.path.join("data", f"processed_{args.symbol}.csv")
    model_path = os.path.join("model", f"{args.symbol}_rf.joblib")

    if not os.path.exists(processed_path):
        raise FileNotFoundError("Processed data not found. Run preprocess.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    df = pd.read_csv(processed_path)
    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_") or c.startswith("ret_")]

    model = joblib.load(model_path)

    # Predict for last available row
    latest_data = df.iloc[-args.days:][feature_cols]
    preds = model.predict(latest_data)

    for i, pred in enumerate(preds, 1):
        print(f"Day +{i} predicted close: {pred:.2f}")
