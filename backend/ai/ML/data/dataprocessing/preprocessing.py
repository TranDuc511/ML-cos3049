import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


# Columns that get scaled (must match training exactly)
COLS_TO_SCALE = [
    'Transaction amount',
    'Account balance',
    'Salary (per month)',
    'Age',
    'Balance_to_Salary_Ratio',
    'Transaction_to_Balance_Ratio',
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features from raw columns."""

    # Age from Date of Birth
    if 'Date of Birth' in df.columns:
        df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce')
        df['Age'] = (datetime(2026, 3, 6) - df['Date of Birth']).dt.days // 365
        df = df.drop(columns=['Date of Birth'])

    # Time flags from Timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Hour']      = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    # Weekend / night flags
    if 'DayOfWeek' in df.columns:
        df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    if 'Hour' in df.columns:
        df['Is_Night'] = df['Hour'].apply(lambda h: 1 if h >= 22 or h <= 6 else 0)

    # Financial ratio features
    if 'Account balance' in df.columns and 'Salary (per month)' in df.columns:
        salary = df['Salary (per month)'].replace(0, np.nan)
        df['Balance_to_Salary_Ratio'] = df['Account balance'] / salary

    if 'Transaction amount' in df.columns and 'Account balance' in df.columns:
        balance = df['Account balance'].replace(0, np.nan)
        df['Transaction_to_Balance_Ratio'] = df['Transaction amount'] / balance

    df = df.fillna(0)
    return df


def normalize(df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """
    Scale numeric columns with MinMaxScaler.
    Saves the fitted scaler to save_path so it can be reused at inference time.
    """
    available = [c for c in COLS_TO_SCALE if c in df.columns]

    scaler = MinMaxScaler()
    df[available] = scaler.fit_transform(df[available])

    # Save the scaler — REQUIRED for consistent inference
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to: {save_path}")

    return df


if __name__ == '__main__':
    HERE        = os.path.dirname(__file__)
    INPUT_FILE  = os.path.join(HERE, '..', 'data_2', 'data_encoded.json')
    OUTPUT_FILE = os.path.join(HERE, '..', 'data_2', 'data_processed.json')
    SCALER_PATH = os.path.join(HERE, '..', '..', 'models', 'scaler.pkl')

    df = pd.read_json(INPUT_FILE)
    df = extract_features(df)
    df = normalize(df, save_path=SCALER_PATH)

    df.to_json(OUTPUT_FILE, orient='records', indent=4)
    print(f'Done. Saved {len(df):,} rows -> {OUTPUT_FILE}')
    print(f'Columns: {list(df.columns)}')