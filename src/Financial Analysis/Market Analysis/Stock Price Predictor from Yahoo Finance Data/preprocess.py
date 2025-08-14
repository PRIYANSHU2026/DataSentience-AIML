# preprocess.py

import pandas as pd

def parse_numeric(value):
    if isinstance(value, str):
        value = value.strip().replace(',', '')
        multiplier = 1
        if value.endswith('M'):
            multiplier = 1e6
            value = value[:-1]
        elif value.endswith('B'):
            multiplier = 1e9
            value = value[:-1]
        elif value.endswith('T'):
            multiplier = 1e12
            value = value[:-1]
        try:
            return float(value) * multiplier
        except:
            return None
    return value

def preprocess_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)

    # Clean columns
    df['change %'] = df['change %'].str.replace('%', '').str.replace('+', '').astype(float)
    df['volume'] = df['volume'].apply(parse_numeric)
    df['market_cap'] = df['market_cap'].apply(parse_numeric)
    df['pe_ratio'] = pd.to_numeric(df['pe_ratio'].replace('--', None), errors='coerce')

    # Drop rows with missing target
    df = df.dropna(subset=['price'])

    # Drop non-numeric columns
    df = df.drop(columns=['symbol', 'name'])

    # Drop remaining NaNs
    df = df.dropna()

    return df
