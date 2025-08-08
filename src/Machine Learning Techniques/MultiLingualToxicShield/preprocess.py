"""
preprocess.py

- Loads data/datafile (default: data/LLM__data.csv)
- Drops unused columns
- Cleans missing values
- Normalizes timestamp and creates `combined_text` = text + " " + response
- Writes processed data to data/processed_LLM_data.csv
"""

import os
import pandas as pd

DEFAULT_INPUT = os.path.join("data", "LLM__data.csv")
DEFAULT_OUTPUT = os.path.join("data", "processed_LLM_data.csv")


def load_data(path: str = DEFAULT_INPUT) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put your dataset at {path}")
    df = pd.read_csv(path)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop index-like unnamed column if present
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Ensure expected columns exist
    expected = {"from_language", "model", "time", "text", "response"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Drop rows where both text and response are missing (or either, depending on preference)
    df = df.dropna(subset=["text", "response"], how="any").reset_index(drop=True)

    # Normalize time column to datetime
    try:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    except Exception:
        df["time"] = pd.to_datetime(df["time"].astype(str), errors="coerce")

    # Create combined text for toxicity detection (prompt + response)
    df["combined_text"] = df["text"].astype(str).str.strip() + " " + df["response"].astype(str).str.strip()

    # Optional: add simple length features
    df["prompt_len"] = df["text"].astype(str).apply(len)
    df["response_len"] = df["response"].astype(str).apply(len)
    df["combined_len"] = df["combined_text"].apply(len)

    return df


def save_processed(df: pd.DataFrame, out_path: str = DEFAULT_OUTPUT):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[preprocess] Processed data saved to {out_path}")


def main(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT):
    print("[preprocess] Loading data...")
    df = load_data(input_path)
    print(f"[preprocess] Raw rows: {len(df)}")
    df_clean = clean_dataframe(df)
    print(f"[preprocess] Cleaned rows: {len(df_clean)}")
    save_processed(df_clean, output_path)


if __name__ == "__main__":
    main()
