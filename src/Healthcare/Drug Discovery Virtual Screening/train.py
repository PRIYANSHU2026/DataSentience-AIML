import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data

MODEL_PATH = "model/drug_classifier.pkl"

if __name__ == "__main__":
    # Load and split dataset
    X_train, X_test, y_train, y_test = load_data("data/drug_discovery_virtual_screening.csv")

    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(exclude=['int64', 'float64']).columns

    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Create full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Save model
    joblib.dump(pipeline, MODEL_PATH)

    # Test model
    acc = pipeline.score(X_test, y_test)
    print(f"Model trained. Test Accuracy: {acc:.4f}")
