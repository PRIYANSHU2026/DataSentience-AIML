import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from catboost import CatBoostClassifier

from preprocess import load_and_preprocess_data

# Paths
DATA_PATH = 'data/airbnb_nyc.csv'
MODEL_PATH = 'model/booking_model.pkl'

def train_model():
    # Load data
    X, y, preprocessor = load_and_preprocess_data(DATA_PATH)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify categorical feature indices (IMPORTANT for CatBoost)
    cat_features = ['neighbourhood_group', 'room_type']
    cat_indices = [X.columns.get_loc(col) for col in cat_features]

    # Model
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100
    )

    # Train (CatBoost uses raw X, NOT sklearn pipeline preprocessing here)
    model.fit(
        X_train, y_train,
        cat_features=cat_indices,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()