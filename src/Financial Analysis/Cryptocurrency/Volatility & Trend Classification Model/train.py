"""
train.py
- Loads processed CSV produced by preprocess.py
- Encodes labels and trains a classifier (RandomForest by default)
- Saves model, label encoder and metrics to model/
"""
import pandas as pd
import argparse
import os
import joblib
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Symbol to train on (e.g., BTCUSDT)")
    parser.add_argument("--model-name", default="rf", choices=["rf"], help="Which model to train")
    args = parser.parse_args()

    processed_path = os.path.join("data", f"processed_{args.symbol}_classified.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed CSV not found at {processed_path}. Run preprocess.py first.")

    df = pd.read_csv(processed_path)
    # Choose features: numeric columns except label / date / symbol
    ignore_cols = {'date','symbol','market_state','trend_label','vol_label'}
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for training.")

    X = df[feature_cols]
    y = df['market_state']

    # Time-based split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    if args.model_name == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise NotImplementedError("Only rf implemented in this script.")

    model.fit(X_train, y_train_enc)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    report = classification_report(y_test_enc, preds, target_names=le.classes_, digits=4)
    cm = confusion_matrix(y_test_enc, preds).tolist()

    # Save artifacts
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", f"{args.symbol}_marketstate_{args.model_name}.joblib")
    le_path = os.path.join("model", f"{args.symbol}_label_encoder.joblib")
    metrics_path = os.path.join("model", f"{args.symbol}_metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(le, le_path)

    metrics = {
        "accuracy": float(acc),
        "classes": le.classes_.tolist(),
        "classification_report": report,
        "confusion_matrix": cm
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Label encoder saved to: {le_path}")
    print(f"📊 Accuracy: {acc:.4f}")
    print("Classification report:\n", report)
