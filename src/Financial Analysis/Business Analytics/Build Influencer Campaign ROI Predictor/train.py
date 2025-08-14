# train.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import preprocess_data

# Ensure model dir exists
os.makedirs("model", exist_ok=True)

# --- Load dataset ---
DATA_PATH = "data/influencer_marketing_roi_dataset.csv"
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# --- Preprocess (returns X, y, preprocessor) ---
X, y, preprocessor = preprocess_data(df)

# --- Drop rows with missing target or missing features ---
mask_valid = y.notna() & (~X.isnull().any(axis=1))
X = X[mask_valid].reset_index(drop=True)
y = y[mask_valid].reset_index(drop=True)

print(f"Rows after dropping invalid/missing: {len(X)}")

# --- Train / Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

# --- Build pipeline and train ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)

print("Training pipeline...")
pipeline.fit(X_train, y_train)
print("Training completed.")

# --- Predict on test set ---
y_pred = pipeline.predict(X_test)

# --- Robust numeric conversion & sync-cleaning ---
y_test_arr = pd.to_numeric(pd.Series(y_test), errors="coerce").to_numpy(dtype=float)
y_pred_arr = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float)

# Keep only rows where both are finite numbers
valid_mask = np.isfinite(y_test_arr) & np.isfinite(y_pred_arr)
y_test_clean = y_test_arr[valid_mask]
y_pred_clean = y_pred_arr[valid_mask]

print(f"Valid rows for metrics: {len(y_test_clean)}")

# --- Metrics (version-agnostic: compute RMSE via sqrt of MSE) ---
if len(y_test_clean) > 0:
    mse = mean_squared_error(y_test_clean, y_pred_clean)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test_clean, y_pred_clean)
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
else:
    print("⚠️ No valid numeric rows to evaluate metrics. Check data cleaning step.")

# --- Save pipeline (preprocessor + model) ---
MODEL_PATH = "model/roi_predictor.pkl"
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Pipeline saved to {MODEL_PATH}")
