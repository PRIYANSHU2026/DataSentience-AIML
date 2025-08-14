import pandas as pd
import argparse
import os

def create_features(df, lags=[1,2,3,5,7,14], roll_windows=[7,30,60]):
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["close_next"] = df["close"].shift(-1)  # target

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["close"].shift(lag)

    # Rolling stats
    for w in roll_windows:
        df[f"roll_mean_{w}"] = df["close"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["close"].rolling(w).std()

    # Returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_7"] = df["close"].pct_change(7)

    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to preprocess (e.g., BTCUSDT)")
    args = parser.parse_args()

    data_path = os.path.join("data", "top_100_cryptos_with_correct_network.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV not found at {data_path}")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["symbol"] == args.symbol].sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No data found for symbol {args.symbol}")

    processed_df = create_features(df)
    save_path = os.path.join("data", f"processed_{args.symbol}.csv")
    processed_df.to_csv(save_path, index=False)

    print(f"✅ Processed data saved to {save_path} ({processed_df.shape[0]} rows)")
