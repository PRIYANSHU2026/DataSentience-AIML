from prophet import Prophet
import joblib
from preprocess import load_and_preprocess

def train_model():
    df = load_and_preprocess("data/exchange_data.csv")

    model = Prophet()
    model.fit(df)

    # Save model (optional)
    joblib.dump(model, "models/prophet_model.pkl")
    print("✅ Model trained and saved.")

if __name__ == "__main__":
    train_model()
