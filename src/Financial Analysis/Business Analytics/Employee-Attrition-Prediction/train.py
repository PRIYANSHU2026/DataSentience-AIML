# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("data/employee_attrition.csv")

# Drop non-useful columns
df.drop(['EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours'], axis=1, inplace=True)

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categoricals
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].apply(lambda col: col.astype('category').cat.codes)

# Handle missing values (if any)
df.fillna(0, inplace=True)

# Split features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model
model = CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/catboost_model.pkl")

print("✅ Model trained and saved to models/catboost_model.pkl")
