import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['reviews_per_month'])

    # Fill missing categorical values
    df['neighbourhood_group'] = df['neighbourhood_group'].fillna("Unknown")
    df['room_type'] = df['room_type'].fillna("Unknown")

    # 2. Smart imputation (grouped median)

    if 'price' in df.columns:
        df['price'] = df['price'].fillna(
            df.groupby('neighbourhood_group')['price'].transform('median')
        )

    # fallback for any remaining nulls
    df['price'] = df['price'].fillna(df['price'].median())


    # Log transform (fix skewed price distribution)
    df['price_log'] = np.log1p(df['price'])

    # Binary target
    df['is_frequently_booked'] = (df['reviews_per_month'] > 1).astype(int)
    features = [
        'neighbourhood_group',
        'room_type',
        'price_log',
        'minimum_nights',
        'number_of_reviews',
        'availability_365',
        'calculated_host_listings_count'
    ]

    target = 'is_frequently_booked'

    X = df[features]
    y = df[target]


    cat_features = ['neighbourhood_group', 'room_type']
    num_features = [col for col in features if col not in cat_features]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_features),
        ('cat', categorical_pipeline, cat_features)
    ])

    return X, y, preprocessor