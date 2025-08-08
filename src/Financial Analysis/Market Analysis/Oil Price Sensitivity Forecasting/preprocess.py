import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df = df.dropna()

    # Select features and target
    features = [
        'USD_ExchangeRate',
        'Policy_Rate(%)',
        '10Y_Bond_Yield(%)',
        'Yield_Spread(10Y_vs_US)(%)',
        'Equity_Index_Level'
    ]
    target = 'Oil_Price(USD_per_bbl)'

    X = df[features]
    y = df[target]

    return X, y
