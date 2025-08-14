import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str):
    """
    Loads dataset, removes ID columns, splits into train/test.
    """
    df = pd.read_csv(path)

    # Drop any ID-like columns (adjust if your file uses a different name)
    for col in df.columns:
        if "id" in col.lower() or "cid" in col.lower():
            df = df.drop(columns=[col])

    # Separate features and target
    if 'active' not in df.columns:
        raise ValueError("Target column 'active' not found in dataset")
    
    X = df.drop(columns=['active'])
    y = df['active']

    # Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
