# train.py
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess import load_and_preprocess

if __name__ == "__main__":
    data_path = "data/ai_financial_market_daily_realistic_synthetic.csv"

    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(data_path)

    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

    # Save model
    joblib.dump(model, "model/trained_model.pkl")
    print("Model saved to model/trained_model.pkl")
