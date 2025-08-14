"""
predict.py
- Loads trained classifier + label encoder
- Loads latest rows from processed CSV
- Predicts market_state for the last N rows or next day based on last available features
- Prints probabilities for each class and saves a simple CSV of results
"""
import pandas as pd
import argparse
import os
import joblib
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Symbol to predict for (e.g., BTCUSDT)")
    parser.add_argument("--rows", type=int, default=7, help="Number of recent rows to predict (default 7)")
    args = parser.parse_args()

    processed_path = os.path.join("data", f"processed_{args.symbol}_classified.csv")
    model_path_pattern = os.path.join("model", f"{args.symbol}_marketstate_*.joblib")

    if not os.path.exists(processed_path):
        raise FileNotFoundError("Processed data not found. Run preprocess.py first.")
    # find the model file (there should be only one for the symbol in this simple setup)
    import glob
    model_files = glob.glob(os.path.join("model", f"{args.symbol}_marketstate_*.joblib"))
    if not model_files:
        raise FileNotFoundError("Model not found. Run train.py first.")
    model_path = model_files[0]
    le_path = os.path.join("model", f"{args.symbol}_label_encoder.joblib")
    if not os.path.exists(le_path):
        raise FileNotFoundError("Label encoder not found. Run train.py first.")

    df = pd.read_csv(processed_path)
    df = df.sort_values("date").reset_index(drop=True)

    # features selection: numeric features used in training (exclude labels etc.)
    ignore_cols = {'date','symbol','market_state','trend_label','vol_label'}
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Load model & encoder
    model = joblib.load(model_path)
    le = joblib.load(le_path)

    # Select last rows
    rows = args.rows
    latest = df.iloc[-rows:].copy()
    X_latest = latest[feature_cols].values

    preds_enc = model.predict(X_latest)
    preds_proba = None
    if hasattr(model, "predict_proba"):
        preds_proba = model.predict_proba(X_latest)

    preds = le.inverse_transform(preds_enc)

    out = latest[['date','close']].reset_index(drop=True)
    out['predicted_state'] = preds
    if preds_proba is not None:
        # create columns for each class probability
        classes = le.classes_
        for i, cls in enumerate(classes):
            out[f'prob_{cls}'] = preds_proba[:, i]

    out_path = os.path.join("model", f"{args.symbol}_predictions_recent.csv")
    out.to_csv(out_path, index=False)

    print("Recent predictions (saved to):", out_path)
    print(out.to_string(index=False))
