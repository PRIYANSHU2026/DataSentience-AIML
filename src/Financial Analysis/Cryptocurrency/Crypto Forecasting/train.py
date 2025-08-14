import pandas as pd
import argparse
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to train on (e.g., BTCUSDT)")
    args = parser.parse_args()

    processed_path = os.path.join("data", f"processed_{args.symbol}.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed CSV not found. Run preprocess.py first.")

    df = pd.read_csv(processed_path)
    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_") or c.startswith("ret_")]
    target_col = "close_next"

    # Train/test split by time
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", f"{args.symbol}_rf.joblib")
    joblib.dump(model, model_path)

    metrics = {"MAE": mae, "RMSE": rmse}
    with open(os.path.join("model", f"{args.symbol}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Model trained and saved to {model_path}")
    print(f"📊 Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}")
