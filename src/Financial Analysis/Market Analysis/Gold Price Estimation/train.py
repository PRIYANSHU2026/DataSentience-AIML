import pandas as pd
import xgboost as xgb
import joblib
from preprocess import preprocess_data

# Load data
df = pd.read_csv("data/gold_price.csv")

# Preprocess data
X, y = preprocess_data(df)

# Train XGBoost Regressor
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X, y)

# Save model
joblib.dump(model, "model/gold_price_model.pkl")
print("Model training complete and saved.")