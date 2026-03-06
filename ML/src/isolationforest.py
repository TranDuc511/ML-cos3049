##IsolationForest Clustering

import json
import pandas as pd
from sklearn.ensemble import IsolationForest
from encoding import encode_columns

import matplotlib.pyplot as plt
import seaborn as sns

# print to terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def load_data(file_path):

    print(f"STEP 1 Reading data from: {file_path}")
    data_frame = pd.read_json(file_path) 
    print(f"Uploaded {len(data_frame)} data.")
    return data_frame

def clean_and_process_data(data_frame):
    print("Step 2 Processing data")
    
    #1. Clean the number columns (remove commas and convert to numbers)
    numeric_columns = ['Transaction amount', 'Account balance', 'Salary (per month)']
    for column in numeric_columns:
        # Chuyển thành chuỗi (string) để xử lý
        data_frame[column] = data_frame[column].astype(str)
        
        # Remove comma and extra decimal points
        data_frame[column] = data_frame[column].str.replace(',', '', regex=False)
        data_frame[column] = data_frame[column].str.replace('.', '', regex=False)
        
        # Convert to number (if error then become 0)
        data_frame[column] = pd.to_numeric(data_frame[column], errors='coerce').fillna(0)

    #2. Time processing
    if 'Timestamp' in data_frame.columns:
        # Convert timestamp to datetime
        data_frame['DateTime'] = pd.to_datetime(data_frame['Timestamp'], unit='ms')
        
        # Create additional columns for Hour and Day of Week
        data_frame['Hour'] = data_frame['DateTime'].dt.hour
        data_frame['DayOfWeek'] = data_frame['DateTime'].dt.dayofweek
    
    #3. Label Encoding
    # Machine only understand number, so we need to convert text columns to numbers
    text_columns = [
        'Transaction Detail', 'Geological', 'Device Use', 
        'Gender', 'Location', 'Working Status'
    ]
    
    # Create a copy to train (only contains numbers)
    data_frame_processed = data_frame.copy()
    
    # Encode text columns and add them with _encoded suffix
    available_text_cols = [col for col in text_columns if col in data_frame.columns]
    encoded_df = data_frame[available_text_cols].copy()
    encoded_df, _ = encode_columns(encoded_df, columns=available_text_cols)
    for col in available_text_cols:
        data_frame_processed[col + "_encoded"] = encoded_df[col]

    return data_frame, data_frame_processed

def detect_anomalies(data_frame, data_frame_processed):
    print("Step 3 Detecting Anomalies...")
    
    # Select features to train the model (only select numeric columns)
    features = [
        'Transaction amount', 'Account balance', 'Salary (per month)',
        'Hour', 'DayOfWeek',
        'Transaction Detail_encoded', 'Geological_encoded', 
        'Device Use_encoded', 'Location_encoded', 'Working Status_encoded'
    ]
    
    # Filter only columns that exist in the data
    selected_features = [col for col in features if col in data_frame_processed.columns]
    
    # Create model input data (Features)
    X = data_frame_processed[selected_features]
    
    # Initialize Isolation Forest
    # contamination=0.15 means we predict that 15% of the data is anomalous
    model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    
    # Train model
    model.fit(X)
    
    # Predict (-1 is Anomaly, 1 is Normal)
    predictions = model.predict(X)
    
    # Assign labels to the original data: 1 = Fraud, 0 = Normal
    # If prediction is -1 then it is Fraud (1), otherwise it is 0
    data_frame['is_fraud'] = [1 if p == -1 else 0 for p in predictions]
    
    # Calculate anomaly score (Lower score means more anomalous)
    data_frame['anomaly_score'] = model.score_samples(X)
    
    fraud_count = data_frame['is_fraud'].sum()
    print(f"Found {fraud_count} suspicious transactions.")
    
    return data_frame

def save_result(data_frame, output_path):
    """
    Step 4: Save Result
    """
    print(f"Step 4 Saving result to file: {output_path}")
    
    # Convert datetime column to string for JSON saving
    if 'DateTime' in data_frame.columns:
        data_frame['DateTime'] = data_frame['DateTime'].astype(str)
        
    data_frame.to_json(output_path, orient='records', indent=4, force_ascii=False)
    print("Done!")

def print_top_anomalies(data_frame):

    print("TOP 5 ANOMALIES")
    
    # Filter Fraud transactions and sort by anomaly score (lower is more different)
    frauds = data_frame[data_frame['is_fraud'] == 1].sort_values('anomaly_score')
    
    # Show important columns
    columns_to_show = [
        'Transaction ID', 'Transaction amount', 'Transaction Detail', 
        'Location', 'anomaly_score'
    ]
    
   
    print(frauds[columns_to_show].head(5).to_string(index=False))

def visualization(df):
    # 1. Anomaly Score Distribution
    plt.figure()
    plt.hist(df['anomaly_score'], bins=50, color='steelblue')
    plt.axvline(df[df['is_fraud']==1]['anomaly_score'].max(), color='red', linestyle='--', label='Fraud threshold')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.legend()

    # 2. Normal vs Fraud Count
    plt.figure()
    df['is_fraud'].value_counts().rename({0: 'Normal', 1: 'Fraud'}).plot(kind='bar', color=['green', 'red'])
    plt.title('Normal vs Fraud Count')

    # 3. Transaction Amount vs Account Balance
    plt.figure()
    plt.scatter(df['Transaction amount'], df['Account balance'], c=df['is_fraud'], cmap='coolwarm', alpha=0.5)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Account Balance')
    plt.title('Transaction Amount vs Account Balance')

    # 4. Fraud Heatmap by Hour and Day of Week
    plt.figure()
    pivot = df[df['is_fraud']==1].groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)
    sns.heatmap(pivot, cmap='Reds')
    plt.title('Fraud Heatmap: Hour vs Day of Week')

    plt.show()


if __name__ == "__main__":
  
    input_file = 'ML/data/data.json'
    output_file = 'ML/data/data_labeled.json'
    df = load_data(input_file)
        
    df, df_processed = clean_and_process_data(df)
        
    df = detect_anomalies(df, df_processed)
        
    print_top_anomalies(df)
        
    save_result(df, output_file)

    visualization(df)
        
    print("DONE")
        
  
