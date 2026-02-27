"""
CHƯƠNG TRÌNH PHÂN LOẠI GIAO DỊCH BẤT THƯỜNG VỚI RANDOM FOREST
==============================================================

MỤC TIÊU:
---------
1. Đọc dữ liệu đã gán nhãn từ `data/data_labeled.json`.
2. Dùng Random Forest để học.
3. In kết quả.

LƯU Ý:
------
Code đơn giản, dễ đọc, không viết tắt.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional

# Cấu hình hiển thị pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Step 1: Đọc dữ liệu từ file JSON."""
    print(f"\n[Step 1] Đọc dữ liệu từ: {file_path}")
    try:
        data_frame = pd.read_json(file_path)
        print(f"   -> Đã tải {len(data_frame)} dòng dữ liệu.")
        return data_frame
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None

def prepare_data(data_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Step 2: Chuẩn bị dữ liệu đầu vào và nhãn để huấn luyện."""
    print("\n[Step 2] Chuẩn bị dữ liệu...")
    
    # 2.1. Chọn các cột đặc trưng (Features)
    feature_columns = [
        'Transaction amount', 'Account balance', 'Salary (per month)', 
        'Hour', 'DayOfWeek', 'Age', 
        'Transaction Detail', 'Geological', 'Device Use', 'Location', 'Working Status'
    ]
    
    # Lọc lấy những cột thực sự tồn tại trong dữ liệu
    available_columns = [col for col in feature_columns if col in data_frame.columns]
    
    features = data_frame[available_columns].copy()
    target = data_frame['is_fraud']
    
    # 2.2. Mã hóa dữ liệu phân loại (chuỗi) thành số
    text_columns = features.select_dtypes(include=['object']).columns
    if len(text_columns) > 0:
        print(f"   -> Đang mã hóa các cột: {list(text_columns)}")
        for column in text_columns:
            encoder = LabelEncoder()
            features[column] = encoder.fit_transform(features[column].astype(str))
        
    return features, target, available_columns

def train_model(features: pd.DataFrame, target: pd.Series, feature_names: List[str]) -> None:
    """Step 3 & 4: Huấn luyện và Đánh giá mô hình."""
    print("\n[Step 3] Training Model...")
    
    # Chia dữ liệu: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Khởi tạo và huấn luyện mô hình Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("   -> Done!")
    
    print("\n[Step 4] Evaluating Model...")
    
    # Dự đoán và tính độ chính xác trên tập kiểm tra
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"   -> Accuracy: {accuracy:.2%} (Tốt nếu > 80%)")
    
    print("\n   -> Bảng chi tiết:")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Fraud']))

    print("\n[Step 5] Yếu tố quan trọng nhất ảnh hưởng đến kết quả")
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    for i in range(min(5, len(feature_names))):
        index = sorted_indices[i]
        print(f"   {i+1}. {feature_names[index]}: {importances[index]:.4f}")

if __name__ == "__main__":
    FILE_PATH = 'ML/data/data_labeled.json'
    
    df = load_data(FILE_PATH)
    if df is not None:
        X, y, columns = prepare_data(df)
        train_model(X, y, columns)
        print("\n[Hoàn tất quá trình]")