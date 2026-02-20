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

# Hiển thị đẹp hơn
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_data(file_path):
    """
    Step 1: Đọc dữ liệu
    """
    print(f"\n[Step 1] Đọc dữ liệu từ: {file_path}")
    try:
        data_frame = pd.read_json(file_path)
        print(f"   -> Đã tải {len(data_frame)} dòng dữ liệu.")
        return data_frame
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None

def prepare_data(data_frame):
    """
    Step 2: Chuẩn bị dữ liệu để học
    """
    print("\n[Step 2] Chuẩn bị dữ liệu...")
    
    # 2.1. Chọn các cột đặc trưng (Features)
    feature_columns = [
        'Transaction amount', 'Account balance', 'Salary (per month)', 
        'Hour', 'DayOfWeek', 'Age', 
        'Transaction Detail', 'Geological', 'Device Use', 'Location', 'Working Status'
    ]
    
    # Lọc lấy cột nào có trong dữ liệu
    available_columns = [col for col in feature_columns if col in data_frame.columns]
    
    # Tạo bảng dữ liệu đầu vào (Input Features - X)
    features = data_frame[available_columns].copy()
    
    # Tạo cột kết quả (Target - y)
    target = data_frame['is_fraud']
    
    # 2.2. Mã hóa dữ liệu chữ thành số
    # Máy học cần số, nên ta chuyển hết chữ sang số
    text_columns = features.select_dtypes(include=['object']).columns
    print(f"   -> Đang mã hóa các cột: {list(text_columns)}")
    
    for column in text_columns:
        encoder = LabelEncoder()
        # Chuyển chữ thành số (0, 1, 2...)
        features[column] = encoder.fit_transform(features[column].astype(str))
        
    return features, target, available_columns

def train_model(features, target, feature_names):
    """
    Step 3: Train and Evaluate
    """
    print("\n[Step 3] Traing Model...")
    
    # Data Splitting: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Khởi tạo mô hình Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Bắt đầu học
    model.fit(X_train, y_train)
    print("   -> Done!")
    
    print("\n[Step 4] Evaluating Model...")
    
    # Dự đoán trên tập kiểm tra
    predictions = model.predict(X_test)
    
    # Tính độ chính xác
    accuracy = accuracy_score(y_test, predictions)
    print(f"   -> Accuracy: {accuracy:.2%} (Tốt nếu > 80%)")
    
    print("\n   -> Bảng chi tiết:")
    # In bảng báo cáo (Precision, Recall, F1-score)
    print(classification_report(y_test, predictions, target_names=['Normal', 'Fraud']))

    # Xem tính năng nào quan trọng nhất
    print("\n[Step 5] Yếu tố quan trọng nhất ảnh hưởng đến kết quả")
    importances = model.feature_importances_
    
    # Sắp xếp giảm dần
    sorted_indices = np.argsort(importances)[::-1]
    
    for i in range(min(5, len(feature_names))):
        index = sorted_indices[i]
        feature_name = feature_names[index]
        importance_score = importances[index]
        print(f"   {i+1}. {feature_name}: {importance_score:.4f}")

if __name__ == "__main__":
    file_path = 'ML/data/data_labeled.json'
    
    # 1. Đọc dữ liệu
    df = load_data(file_path)
    
    if df is not None:
        # 2. Chuẩn bị
        X, y, columns = prepare_data(df)
        
        # 3. Huấn luyện
        train_model(X, y, columns)
        
        print("DONE")