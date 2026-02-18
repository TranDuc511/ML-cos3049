"""
CHƯƠNG TRÌNH PHÁT HIỆN BẤT THƯỜNG VÀ GẮN NHÃN TỰ ĐỘNG


MỤC TIÊU:
---------
1. Đọc dữ liệu giao dịch từ file `data/data.json`.
2. Tiền xử lý dữ liệu.
3. Sử dụng Isolation Forest để phát hiện giao dịch bất thường.
4. Gắn nhãn và lưu kết quả.

LƯU Ý:
------
Code đã được đơn giản hóa và không viết tắt để dễ đọc.
"""

import json
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# print to terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def load_data(file_path):

    """
    Step 1: Read from Json
    """
    print(f"\n[STEP 1] Reading data from: {file_path}")
    try:
        # using pandas to read
        data_frame = pd.read_json(file_path)
    except ValueError:
        # if wrong json format, read manualy and switch to dataframe
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        data_frame = pd.DataFrame(data)
    
    print(f"   -> uploaded {len(data_frame)} data.")
    return data_frame

def clean_and_process_data(data_frame):
    """
    Step 2: Làm sạch và chuẩn bị dữ liệu
    """
    print("\n[Step 2] Đang xử lý dữ liệu...")
    
    # 2.1. Làm sạch các cột số (loại bỏ dấu phẩy và chuyển thành số)
    numeric_columns = ['Transaction amount', 'Account balance', 'Salary (per month)']
    for column in numeric_columns:
        # Chuyển thành chuỗi (string) để xử lý
        data_frame[column] = data_frame[column].astype(str)
        
        # Xóa dấu phẩy và dấu chấm thừa
        data_frame[column] = data_frame[column].str.replace(',', '', regex=False)
        data_frame[column] = data_frame[column].str.replace('.', '', regex=False)
        
        # Chuyển thành số (nếu lỗi thì thành 0)
        data_frame[column] = pd.to_numeric(data_frame[column], errors='coerce').fillna(0)

    # 2.2. Xử lý thời gian
    if 'Timestamp' in data_frame.columns:
        # Chuyển đổi timestamp sang định dạng ngày giờ
        data_frame['DateTime'] = pd.to_datetime(data_frame['Timestamp'], unit='ms')
        
        # Tạo thêm cột Giờ và Thứ trong tuần
        data_frame['Hour'] = data_frame['DateTime'].dt.hour
        data_frame['DayOfWeek'] = data_frame['DateTime'].dt.dayofweek
    
    # 2.3. Label Encoding
    # Machine only understand number, so we need to convert text columns to numbers
    text_columns = [
        'Transaction Detail', 'Geological', 'Device Use', 
        'Gender', 'Location', 'Working Status'
    ]
    
    # Create a copy to train (only contains numbers)
    data_frame_processed = data_frame.copy()
    
    for column in text_columns:
        if column in data_frame.columns:
            encoder = LabelEncoder()
            # Create new column name with _encoded suffix
            new_column_name = column + "_encoded"
            data_frame_processed[new_column_name] = encoder.fit_transform(data_frame[column].astype(str))

    return data_frame, data_frame_processed

def detect_anomalies(data_frame, data_frame_processed):
    """
    Step 3: Detect anomalies using Isolation Forest
    """
    print("\n[Step 3] Đang tìm kiếm giao dịch bất thường...")
    
    # Select features to train the model (only select numeric columns)
    features = [
        'Transaction amount', 'Account balance', 'Salary (per month)',
        'Hour', 'DayOfWeek',
        'Transaction Detail_encoded', 'Geological_encoded', 
        'Device Use_encoded', 'Location_encoded', 'Working Status_encoded'
    ]
    
    # Lọc chỉ lấy các cột thực sự có trong dữ liệu
    selected_features = [col for col in features if col in data_frame_processed.columns]
    
    # Tạo dữ liệu đầu vào cho mô hình (Features)
    X = data_frame_processed[selected_features]
    
    # Khởi tạo thuật toán Isolation Forest
    # contamination=0.05 nghĩa là ta dự đoán có khoảng 5% dữ liệu là bất thường
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    
    # Huấn luyện mô hình
    model.fit(X)
    
    # Dự đoán kết quả (-1 là Bất thường, 1 là Bình thường)
    predictions = model.predict(X)
    
    # Gán nhãn lại vào dữ liệu gốc: 1 = Gian lận (Fraud), 0 = Bình thường
    # Nếu prediction là -1 thì là Fraud (1), ngược lại là 0
    data_frame['is_fraud'] = [1 if p == -1 else 0 for p in predictions]
    
    # Tính điểm bất thường (Score càng thấp càng bất thường)
    data_frame['anomaly_score'] = model.score_samples(X)
    
    fraud_count = data_frame['is_fraud'].sum()
    print(f"   -> Đã tìm thấy {fraud_count} giao dịch đáng ngờ.")
    
    return data_frame

def save_result(data_frame, output_path):
    """
    Step 4: Lưu kết quả
    """
    print(f"\n[Step 4] Lưu kết quả vào file: {output_path}")
    
    # Chuyển cột thời gian về dạng chuỗi để lưu được vào JSON
    if 'DateTime' in data_frame.columns:
        data_frame['DateTime'] = data_frame['DateTime'].astype(str)
        
    data_frame.to_json(output_path, orient='records', indent=4, force_ascii=False)
    print("   -> Đã lưu thành công.")

def print_top_anomalies(data_frame):
    """
    In ra 5 giao dịch bất thường nhất
    """
    print("\n" + "="*80)
    print("TOP 5 GIAO DỊCH ĐÁNG NGỜ NHẤT")
    print("="*80)
    
    # Lọc lấy giao dịch Fraud và sắp xếp theo điểm score tăng dần (càng thấp càng dị biệt)
    frauds = data_frame[data_frame['is_fraud'] == 1].sort_values('anomaly_score')
    
    # Show important columns
    columns_to_show = [
        'Transaction ID', 'Transaction amount', 'Transaction Detail', 
        'Location', 'anomaly_score'
    ]
    
    #print 5 columns
    print(frauds[columns_to_show].head(5).to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    # Đường dẫn file dữ liệu
    input_file = 'ML/data/data.json'
    output_file = 'ML/data/data_labeled.json'
    
    try:
        # 1. Đọc dữ liệu
        df = load_data(input_file)
        
        # 2. Xử lý dữ liệu
        df, df_processed = clean_and_process_data(df)
        
        # 3. Phát hiện bất thường
        df = detect_anomalies(df, df_processed)
        
        # 4. In báo cáo
        print_top_anomalies(df)
        
        # 5. Lưu file
        save_result(df, output_file)
        
        print("DONE")
        
    except FileNotFoundError:
        print(f"Error: Not found {input_file}")
    except Exception as e:
        print(f"Error: {e}")
