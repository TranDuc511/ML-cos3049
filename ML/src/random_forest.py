
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

import json

# Cấu hình hiển thị pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_data(file_path):
    """Step 1: Đọc dữ liệu từ file JSON."""
    
    print(f"Step 1: Reading {file_path}")
    
    data_frame = pd.read_json(file_path)
    print(f"Đã tải {len(data_frame)} dòng dữ liệu.")
    return data_frame


def prepare_data(data_frame):
    
    print("Step 2 Preparing data")
    
    #1.Select the feature columns
    feature_columns = [
        'Transaction amount', 'Account balance', 'Salary (per month)', 
        'Hour', 'DayOfWeek', 
        'Transaction Detail', 'Geological', 'Device Use', 'Location', 'Working Status'
    ]
    
    features = data_frame[feature_columns]
    target = data_frame['is_fraud']
    
    
    #2.Encrypting classified data (strings) into numbers
    text_columns = features.select_dtypes(include=['object']).columns
    if len(text_columns) > 0:
        print(f"Encryting: {list(text_columns)}")
        for column in text_columns:
            encoder = LabelEncoder()
            features[column] = encoder.fit_transform(features[column].astype(str))
        
    return features, target, feature_columns

def train_model(features, target, feature_names):
   
    print("Step 3 Training Model")
    
    # Chia dữ liệu: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
   
    # Khởi tạo và huấn luyện mô hình Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Done")
    
    print("Step 4 Evaluating Model")
    
    # Dự đoán và tính độ chính xác trên tập kiểm tra
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Accuracy: {accuracy:.2%}")
    
    print("Bảng chi tiết:")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Fraud']))

    
    print("Step 5: Important features")
    importance_df = pd.DataFrame({
        'Features': feature_names, 
        'Importances': model.feature_importances_
    }).sort_values('Importances', ascending=False)

    # Hiển thị kết quả
    print(importance_df.head(5))



if __name__ == "__main__":
    FILE_PATH = 'ML/data/data_labeled.json'
    df = load_data(FILE_PATH)
    X, y, columns = prepare_data(df)
    train_model(X, y, columns)
    print("Hoàn tất quá trình")