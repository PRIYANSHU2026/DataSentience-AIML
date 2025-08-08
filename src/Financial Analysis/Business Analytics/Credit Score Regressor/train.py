import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import load_data, preprocess_features, get_target

# Load and prepare data
df = load_data('data/synthetic_personal_finance_dataset.csv')
X, preprocessor = preprocess_features(df)
y = get_target(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit preprocessor
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)

# Evaluate
y_pred = model.predict(X_test_processed)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

# Save artifacts
joblib.dump(preprocessor, 'model/preprocessor.joblib')
joblib.dump(model, 'model/credit_score_model.joblib')
