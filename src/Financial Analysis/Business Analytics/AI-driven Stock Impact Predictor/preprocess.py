# preprocess.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Fill missing Event values
    df['Event'] = df['Event'].fillna("No Event")

    # Convert Date to datetime and extract features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)

    # Features & target
    X = df.drop(columns=['Stock_Impact_%'])
    y = df['Stock_Impact_%']

    # Identify categorical and numerical features
    categorical_features = ['Company', 'Event']
    numerical_features = X.drop(columns=categorical_features).columns

    # Preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessor
    joblib.dump(preprocessor, "model/preprocessor.pkl")

    return X_train, X_test, y_train, y_test, preprocessor
