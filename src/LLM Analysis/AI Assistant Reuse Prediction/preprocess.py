# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Convert date to datetime & extract useful features
    df["SessionDate"] = pd.to_datetime(df["SessionDate"])
    df["SessionMonth"] = df["SessionDate"].dt.month
    df["SessionDay"] = df["SessionDate"].dt.dayofweek

    # Define features and target
    X = df.drop(columns=["SessionID", "SessionDate", "UsedAgain"])
    y = df["UsedAgain"].astype(int)  # Convert bool → int

    # Identify categorical & numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Preprocessing pipelines
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor
