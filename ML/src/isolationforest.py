import json
import pandas as pd
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def load_data(file_path):
    print(f"STEP 1 Reading preprocessed data from: {file_path}")
    df = pd.read_json(file_path) 
    
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    print(f"Loaded {len(df):,} transactions.")
    return df


def detect_anomalies(df):
    print("Step 2 Detecting Anomalies...")
    
    # Select features for the model. We can define exactly what we want,
    # or just use all numeric columns except the IDs and timestamp.
    features = [
        'Transaction amount', 'Account balance', 'Salary (per month)',
        'Hour', 'DayOfWeek', 'Age', 'Is_Weekend', 'Is_Night',
        'Balance_to_Salary_Ratio', 'Tx_to_Balance_Ratio',
        'Transaction Detail', 'Geological', 'Device Use', 
        'Location', 'Working Status', 'Gender', 'Transaction Count'
    ]
    
    selected_features = [col for col in features if col in df.columns]
    X = df[selected_features]
    
    # 15% contamination
    model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    model.fit(X)
    
    predictions = model.predict(X)
    df['is_fraud'] = [1 if p == -1 else 0 for p in predictions]
    df['anomaly_score'] = model.score_samples(X)
    
    print(f"Found {df['is_fraud'].sum()} suspicious transactions.")
    return df


def save_result(df, output_path):
    print(f"Step 3 Saving result to file: {output_path}")
    
    if 'DateTime' in df.columns:
        df['DateTime'] = df['DateTime'].astype(str)
    elif 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].astype(str)
        
    df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    print("Done!")


def print_top_anomalies(df):
    print("TOP 5 ANOMALIES")
    frauds = df[df['is_fraud'] == 1].sort_values('anomaly_score')
    
    columns_to_show = [
        'Transaction ID', 'Transaction amount', 'Transaction Detail', 
        'Location', 'anomaly_score'
    ]
    
    available_cols = [c for c in columns_to_show if c in df.columns]
    print(frauds[available_cols].head(5).to_string(index=False))


def visualization(df):
    # 1. Anomaly Score Distribution
    plt.figure()
    plt.hist(df['anomaly_score'], bins=50, color='steelblue')
    if not df[df['is_fraud']==1].empty:
        plt.axvline(df[df['is_fraud']==1]['anomaly_score'].max(), color='red', linestyle='--', label='Fraud threshold')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.legend()

    # 2. Normal vs Fraud Count
    plt.figure()
    counts = df['is_fraud'].value_counts()
    if 0 in counts: counts = counts.rename({0: 'Normal'})
    if 1 in counts: counts = counts.rename({1: 'Fraud'})
    counts.plot(kind='bar', color=['green', 'red'])
    plt.title('Normal vs Fraud Count')

    # 3. Transaction Amount vs Account Balance
    plt.figure()
    plt.scatter(df['Transaction amount'], df['Account balance'], c=df['is_fraud'], cmap='coolwarm', alpha=0.5)
    plt.xlabel('Transaction Amount (Normalized)')
    plt.ylabel('Account Balance (Normalized)')
    plt.title('Transaction vs Balance')

    # 4. Fraud Heatmap by Hour and Day of Week
    if 'Hour' in df.columns and 'DayOfWeek' in df.columns:
        plt.figure()
        pivot = df[df['is_fraud']==1].groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)
        sns.heatmap(pivot, cmap='Reds')
        plt.title('Fraud Heatmap: Hour vs Day of Week')

    plt.show()


if __name__ == "__main__":
    input_file = 'ML/data/data_processed.json'
    output_file = 'ML/data/data_labeled.json'
    
    df = load_data(input_file)
    df = detect_anomalies(df)
    print_top_anomalies(df)
    save_result(df, output_file)
    visualization(df)
    print("DONE")
