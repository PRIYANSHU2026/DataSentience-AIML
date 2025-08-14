import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_model(data_path, save_path="model/response_quality_model.pkl"):
    df = pd.read_csv(data_path)

    # Features and targets
    X = df[['completeness', 'politeness', 'relevance']]
    y = (0.5 * df['completeness'] + 0.2 * df['politeness'] + 0.3 * df['relevance'])  # Weighted quality score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    joblib.dump(model, save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    train_model("data/LLM__scored_data.csv")