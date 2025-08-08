import pandas as pd

def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses raw gold price data.
    - Converts comma to dot in float fields
    - Drops timestamp columns
    - Converts data types

    Args:
        df (pd.DataFrame): Raw input DataFrame

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
    """
    # Convert comma to dot and cast to float
    for col in ['priceOpen', 'priceHigh', 'priceLow', 'priceClose', 'volume']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)

    # Drop unnecessary timestamp columns
    df = df.drop(columns=['timeOpen', 'timeClose', 'timeHigh', 'timeLow'])

    # Split into features and target
    X = df[['priceOpen', 'priceHigh', 'priceLow', 'volume']]
    y = df['priceClose']

    return X, y