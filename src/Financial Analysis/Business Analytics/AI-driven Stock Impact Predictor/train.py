# train.py
from sklearn.metrics import r2_score
import joblib
from lightgbm import LGBMRegressor
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
        ('regressor', LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    # Save model
    joblib.dump(model, "model/trained_model.pkl")
    print("Model saved to model/trained_model.pkl")
