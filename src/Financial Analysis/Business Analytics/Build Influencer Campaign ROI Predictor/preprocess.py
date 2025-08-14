import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Convert dates with error handling
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')

    # Drop rows with missing start_date after conversion
    df = df.dropna(subset=['start_date'])

    # Extract date features
    df['start_month'] = df['start_date'].dt.month
    df['start_dayofweek'] = df['start_date'].dt.dayofweek

    # Features & target
    X = df[['platform', 'influencer_category', 'campaign_type',
            'engagements', 'estimated_reach', 'campaign_duration_days',
            'start_month', 'start_dayofweek']]
    y = df['product_sales']

    # Preprocessing pipeline
    categorical_features = ['platform', 'influencer_category', 'campaign_type']
    numeric_features = ['engagements', 'estimated_reach', 'campaign_duration_days',
                        'start_month', 'start_dayofweek']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)
        ]
    )
    return X, y, preprocessor
