"""
preprocess.py
- Loads raw CSV from data/
- Filters by symbol
- Creates technical features (lags, rolling means/std, returns, RSI)
- Creates labels for Trend (Bullish/Bearish/Sideways) and Volatility (High/Low)
- Combines them into a single market-state label like "Bullish_HighVol"
- Saves processed CSV to data/processed_<symbol>_classified.csv
"""
import pandas as pd
import numpy as np
import argparse
import os

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def create_features_and_labels(df,
                               trend_window=7,
                               vol_window=30,
                               trend_threshold=0.02,
                               vol_quantile=0.66,
                               lags=[1,2,3,5,7,14],
                               roll_windows=[7,14,30]):
    df = df.copy().sort_values("date").reset_index(drop=True)
    # Basic derived features
    df['range'] = df['high'] - df['low']
    df['intraday_ret'] = (df['close'] - df['open']) / (df['open'] + 1e-9)
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_7'] = df['close'].pct_change(7)
    df['rsi_14'] = rsi(df['close'], period=14)

    # Rolling features
    for w in roll_windows:
        df[f'roll_mean_{w}'] = df['close'].rolling(w).mean()
        df[f'roll_std_{w}'] = df['close'].rolling(w).std()
        df[f'roll_max_{w}'] = df['close'].rolling(w).max()
        df[f'roll_min_{w}'] = df['close'].rolling(w).min()

    # Lags
    for lag in lags:
        df[f'lag_{lag}'] = df['close'].shift(lag)

    # Labels: Trend by percent change over trend_window
    df['trend_ret'] = df['close'].pct_change(periods=trend_window)
    # Trend label rules
    def trend_label(x):
        if x > trend_threshold:
            return 'Bullish'
        elif x < -trend_threshold:
            return 'Bearish'
        else:
            return 'Sideways'
    df['trend_label'] = df['trend_ret'].apply(trend_label)

    # Volatility label: rolling std over vol_window, threshold by quantile
    df['volatility'] = df['close'].pct_change().rolling(vol_window).std()
    # compute threshold on available (non-null) volatility values
    vol_thr = df['volatility'].quantile(vol_quantile)
    df['vol_label'] = np.where(df['volatility'] > vol_thr, 'HighVol', 'LowVol')

    # Combined market state
    df['market_state'] = df['trend_label'] + "_" + df['vol_label']

    # Drop rows with NaN that were caused by rolling/windows/lags
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Symbol to preprocess (e.g., BTCUSDT)")
    parser.add_argument("--data", default=os.path.join("data", "top_100_cryptos_with_correct_network.csv"))
    parser.add_argument("--out", default=None, help="Output processed CSV path (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data CSV not found at: {args.data}")

    df = pd.read_csv(args.data)
    if 'date' not in df.columns:
        raise ValueError("CSV must contain a 'date' column")
    df['date'] = pd.to_datetime(df['date'])

    sub = df[df['symbol'] == args.symbol].sort_values('date').reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No rows found for symbol {args.symbol} in the CSV")

    processed = create_features_and_labels(sub)

    out_path = args.out or os.path.join("data", f"processed_{args.symbol}_classified.csv")
    processed.to_csv(out_path, index=False)
    print(f"✅ Processed and labeled data saved to: {out_path} (rows: {processed.shape[0]})")
