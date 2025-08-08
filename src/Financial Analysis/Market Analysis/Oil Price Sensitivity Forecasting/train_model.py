import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_and_preprocess

def train():
    X, y = load_and_preprocess("data/macro_oil_data.csv")

    model = GradientBoostingRegressor()
    model.fit(X, y)

    # Save model
    joblib.dump(model, "models/oil_price_model.pkl")

    preds = model.predict(X)
    print(f"✅ Model trained.")
    print(f"📉 MSE: {mean_squared_error(y, preds):.4f}")
    print(f"📈 R2 Score: {r2_score(y, preds):.4f}")

if __name__ == "__main__":
    train()
