import pandas as pd

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    # Drop rows with missing values (if any)
    df = df.dropna()

    # Convert 'Date' to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Keep only relevant columns
    df = df[['Date', 'USD_ExchangeRate']]

    # Rename for Prophet compatibility
    df.rename(columns={'Date': 'ds', 'USD_ExchangeRate': 'y'}, inplace=True)

    return df
