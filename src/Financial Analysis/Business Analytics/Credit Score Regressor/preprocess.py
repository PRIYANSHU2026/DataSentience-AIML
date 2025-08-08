import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_features(df):
    X = df.drop(columns=['user_id', 'credit_score', 'record_date'])

    # Define features
    numerical_features = [
        'age', 'monthly_income_usd', 'monthly_expenses_usd', 'savings_usd',
        'loan_amount_usd', 'loan_term_months', 'monthly_emi_usd',
        'loan_interest_rate_pct', 'debt_to_income_ratio', 'savings_to_income_ratio'
    ]

    categorical_features = [
        'gender', 'education_level', 'employment_status',
        'job_title', 'has_loan', 'loan_type', 'region'
    ]

    # Pipelines
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return X, preprocessor

def get_target(df):
    return df['credit_score']
