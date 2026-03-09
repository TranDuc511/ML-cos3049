import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def extract_features(df):
    # Age from Date of Birth
    if 'Date of Birth' in df.columns:
        df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce')
        df['Age'] = (datetime(2026, 3, 6) - df['Date of Birth']).dt.days // 365
        df = df.drop(columns=['Date of Birth'])

    # Time-based flags
    if 'DayOfWeek' in df.columns:
        df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    if 'Hour' in df.columns:
        df['Is_Night'] = df['Hour'].apply(lambda h: 1 if h >= 22 or h <= 6 else 0)

    # Ratio features
    if 'Account balance' in df.columns and 'Salary (per month)' in df.columns:
        salary = df['Salary (per month)'].replace(0, np.nan)
        df['Balance_to_Salary_Ratio'] = df['Account balance'] / salary

    if 'Transaction amount' in df.columns and 'Account balance' in df.columns:
        balance = df['Account balance'].replace(0, np.nan)
        df['Tx_to_Balance_Ratio'] = df['Transaction amount'] / balance

    df = df.fillna(0)
    return df


def normalize(df):
    cols_to_scale = [
        'Transaction amount', 'Account balance', 'Salary (per month)',
        'Age', 'Balance_to_Salary_Ratio', 'Tx_to_Balance_Ratio',
    ]
    available = [c for c in cols_to_scale if c in df.columns]

    scaler = MinMaxScaler()
    df[available] = scaler.fit_transform(df[available])
    return df


if __name__ == '__main__':
    df = pd.read_json('ML/data/data_encoded.json')
    df = extract_features(df)
    df = normalize(df)
    df.to_json('ML/data/data_processed.json', orient='records', indent=4)
    print(f'Done. Saved {len(df):,} rows -> ML/data/data_processed.json')
    print(f'Columns: {list(df.columns)}')
