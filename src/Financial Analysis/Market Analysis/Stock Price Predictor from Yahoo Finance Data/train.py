# train.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess_data

# Load and preprocess
df = preprocess_data("data/yahoo-stocks-data(Uncleaned_data).xlsx")

# Features and target
X = df.drop(columns=["price"])
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")


# Save model
joblib.dump(model, "model/stock_price_predictor.pkl")
