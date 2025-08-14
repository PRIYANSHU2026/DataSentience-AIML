# train.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data
from sklearn.pipeline import Pipeline
DATA_PATH = "data/ai_assistant_usage_student_life.csv"
MODEL_PATH = "model/used_again_predictor.pkl"

# Preprocess
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(DATA_PATH)

# Create pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
