import pandas as pd
import numpy as np

# Clean individual string to number
def extract_number(s):
    if pd.isnull(s):
        return np.nan
    s = str(s).lower().replace(",", "")
    if '-' in s:  # Handle ranges like "70-85 hp"
        parts = [float(p.strip().split()[0]) for p in s.split('-') if p.strip()]
        return np.mean(parts)
    for unit in ['hp', 'km/h', 'sec', 'nm', 'cc']:
        s = s.replace(unit, '')
    try:
        return float(s.strip())
    except:
        return np.nan

# Full preprocessing function
def preprocess_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')


    df['HorsePower_clean'] = df['HorsePower'].apply(extract_number)
    df['TopSpeed_clean'] = df['Total Speed'].apply(extract_number)
    df['Accel_clean'] = df['Performance(0 - 100 )KM/H'].apply(extract_number)

    df.dropna(subset=['HorsePower_clean', 'TopSpeed_clean', 'Accel_clean'], inplace=True)

    return df[['Cars Names', 'HorsePower_clean', 'TopSpeed_clean', 'Accel_clean']]
