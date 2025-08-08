import pandas as pd
import numpy as np

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.dropna()

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(['Country', 'Date'], inplace=True)

    # Compute rolling volatilities
    df['Exchange_Vol'] = df.groupby('Country')['USD_ExchangeRate'].transform(lambda x: x.pct_change().rolling(5).std())
    df['Equity_Vol'] = df.groupby('Country')['Equity_Index_Level'].transform(lambda x: x.pct_change().rolling(5).std())
    df['Yield_Change'] = df.groupby('Country')['Yield_Spread(10Y_vs_US)(%)'].transform(lambda x: x.diff())

    # Custom rule to label risk — you can enhance this later
    def classify_risk(row):
        score = 0
        if row['Exchange_Vol'] > 0.01: score += 1
        if row['Equity_Vol'] > 0.01: score += 1
        if abs(row['Yield_Change']) > 0.1: score += 1
        if row['10Y_Bond_Yield(%)'] > 7.5: score += 1
        if row['Oil_Price(USD_per_bbl)'] > 70: score += 1

        if score <= 1:
            return "Low"
        elif score <= 3:
            return "Medium"
        else:
            return "High"

    df['Risk_Level'] = df.apply(classify_risk, axis=1)
    
    # Features for ML
    features = ['Exchange_Vol', 'Equity_Vol', 'Yield_Change', '10Y_Bond_Yield(%)', 'Oil_Price(USD_per_bbl)']
    X = df[features].dropna()
    y = df.loc[X.index, 'Risk_Level']

    return X, y
