import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import load_and_preprocess

def train():
    X, y = load_and_preprocess("data/country_macro_data.csv")

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, "models/risk_classifier.pkl")
    preds = model.predict(X)

    print("✅ Model Trained. Evaluation:")
    print(classification_report(y, preds))

if __name__ == "__main__":
    train()
