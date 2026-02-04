"""
Isolation Forest - Thuật toán phát hiện giao dịch bất thường (Anomaly Detection)
================================================================================
Thuật toán này sử dụng Isolation Forest để tự động tìm ra những giao dịch "lạc loài" nhất.

Nguyên lý hoạt động:
- Isolation Forest xây dựng nhiều cây quyết định ngẫu nhiên
- Các điểm bất thường (anomalies) dễ dàng bị cô lập hơn trong cây
- Điểm càng bị cô lập nhanh = càng bất thường
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TransactionAnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42):
        """
        Khởi tạo bộ phát hiện giao dịch bất thường
        
        Parameters:
        -----------
        contamination : float (default=0.1)
            Tỷ lệ dữ liệu bất thường dự kiến (10% = 0.1)
        random_state : int
            Seed để đảm bảo kết quả nhất quán
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            verbose=0
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, json_path):
        """Đọc dữ liệu giao dịch từ file JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Chuyển đổi sang DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        print(f"✓ Đã tải {len(df)} giao dịch")
        print(f"✓ Các cột: {df.columns.tolist()}")
        return df
    
    def extract_features(self, df):
        """
        Trích xuất các đặc trưng từ dữ liệu giao dịch
        Tự động phát hiện các cột số và chuyển đổi dữ liệu phù hợp
        """
        features = pd.DataFrame()
        
        # Tự động phát hiện các cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Thêm các cột số trực tiếp
        for col in numeric_cols:
            features[col] = df[col]
        
        # Xử lý cột thời gian nếu có
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        for col in time_cols:
            try:
                # Chuyển đổi timestamp thành giờ trong ngày
                if df[col].dtype == 'object':
                    timestamps = pd.to_datetime(df[col], errors='coerce')
                    features[f'{col}_hour'] = timestamps.dt.hour
                    features[f'{col}_day_of_week'] = timestamps.dt.dayofweek
                elif df[col].dtype in ['int64', 'float64']:
                    # Nếu là Unix timestamp
                    timestamps = pd.to_datetime(df[col], unit='s', errors='coerce')
                    features[f'{col}_hour'] = timestamps.dt.hour
                    features[f'{col}_day_of_week'] = timestamps.dt.dayofweek
            except:
                pass
        
        # Xử lý các cột category (chuyển thành số)
        category_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in category_cols:
            if col not in time_cols:
                # Encode categorical variables
                features[f'{col}_encoded'] = pd.Categorical(df[col]).codes
        
        self.feature_columns = features.columns.tolist()
        print(f"\n✓ Đã trích xuất {len(self.feature_columns)} đặc trưng:")
        for i, col in enumerate(self.feature_columns, 1):
            print(f"  {i}. {col}")
        
        return features
    
    def detect_anomalies(self, df):
        """
        Phát hiện các giao dịch bất thường
        
        Returns:
        --------
        DataFrame với các cột bổ sung:
        - anomaly_score: điểm bất thường (càng âm càng bất thường)
        - is_anomaly: 1 = bất thường, 0 = bình thường
        - anomaly_rank: xếp hạng mức độ bất thường
        """
        # Trích xuất đặc trưng
        features = self.extract_features(df)
        
        # Chuẩn hóa dữ liệu
        features_scaled = self.scaler.fit_transform(features)
        
        # Huấn luyện mô hình và dự đoán
        print("\n🔍 Đang phân tích và tìm kiếm anomalies...")
        predictions = self.model.fit_predict(features_scaled)
        
        # Tính điểm bất thường (anomaly score)
        # Điểm càng âm = càng bất thường
        anomaly_scores = self.model.score_samples(features_scaled)
        
        # Thêm kết quả vào DataFrame gốc
        result_df = df.copy()
        result_df['anomaly_score'] = anomaly_scores
        result_df['is_anomaly'] = (predictions == -1).astype(int)
        
        # Xếp hạng các giao dịch theo mức độ bất thường
        result_df['anomaly_rank'] = result_df['anomaly_score'].rank()
        
        # Thống kê
        n_anomalies = result_df['is_anomaly'].sum()
        print(f"\n✓ Hoàn thành phân tích!")
        print(f"✓ Tìm thấy {n_anomalies} giao dịch bất thường ({n_anomalies/len(df)*100:.1f}%)")
        print(f"✓ {len(df) - n_anomalies} giao dịch bình thường ({(len(df)-n_anomalies)/len(df)*100:.1f}%)")
        
        return result_df
    
    def get_top_anomalies(self, result_df, top_n=10):
        """
        Lấy top N giao dịch bất thường nhất
        """
        anomalies = result_df[result_df['is_anomaly'] == 1].copy()
        anomalies = anomalies.sort_values('anomaly_score')
        
        return anomalies.head(top_n)
    
    def visualize_anomalies(self, result_df, save_path='anomaly_visualization.png'):
        """
        Trực quan hóa kết quả phát hiện anomaly
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Phân tích Giao dịch Bất thường - Isolation Forest', 
                     fontsize=16, fontweight='bold')
        
        # 1. Phân bố điểm anomaly
        ax1 = axes[0, 0]
        ax1.hist(result_df[result_df['is_anomaly']==0]['anomaly_score'], 
                bins=50, alpha=0.7, label='Bình thường', color='green')
        ax1.hist(result_df[result_df['is_anomaly']==1]['anomaly_score'], 
                bins=50, alpha=0.7, label='Bất thường', color='red')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Số lượng giao dịch')
        ax1.set_title('Phân bố Anomaly Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Tỷ lệ anomaly
        ax2 = axes[0, 1]
        counts = result_df['is_anomaly'].value_counts()
        labels = ['Bình thường', 'Bất thường']
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax2.pie(counts.values, labels=labels, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
        ax2.set_title('Tỷ lệ Giao dịch Bất thường')
        
        # 3. Scatter plot của 2 features quan trọng nhất (nếu có đủ features)
        ax3 = axes[1, 0]
        if len(self.feature_columns) >= 2:
            feature_data = self.extract_features(result_df)
            x_col = self.feature_columns[0]
            y_col = self.feature_columns[1] if len(self.feature_columns) > 1 else self.feature_columns[0]
            
            normal = result_df[result_df['is_anomaly'] == 0]
            anomaly = result_df[result_df['is_anomaly'] == 1]
            
            ax3.scatter(feature_data.loc[normal.index, x_col], 
                       feature_data.loc[normal.index, y_col],
                       c='green', alpha=0.5, s=50, label='Bình thường')
            ax3.scatter(feature_data.loc[anomaly.index, x_col], 
                       feature_data.loc[anomaly.index, y_col],
                       c='red', alpha=0.8, s=100, marker='*', label='Bất thường')
            ax3.set_xlabel(x_col)
            ax3.set_ylabel(y_col)
            ax3.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Top 10 giao dịch bất thường nhất
        ax4 = axes[1, 1]
        top_anomalies = self.get_top_anomalies(result_df, top_n=10)
        if len(top_anomalies) > 0:
            y_pos = np.arange(len(top_anomalies))
            scores = top_anomalies['anomaly_score'].values
            ax4.barh(y_pos, scores, color='#e74c3c', alpha=0.7)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([f'#{i+1}' for i in range(len(top_anomalies))])
            ax4.set_xlabel('Anomaly Score')
            ax4.set_title('Top 10 Giao dịch Bất thường Nhất')
            ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Đã lưu biểu đồ: {save_path}")
        plt.show()
    
    def save_results(self, result_df, output_path='anomaly_results.json'):
        """
        Lưu kết quả phân tích ra file JSON
        """
        # Lưu toàn bộ kết quả
        result_df.to_json(output_path, orient='records', indent=2, force_ascii=False)
        print(f"✓ Đã lưu kết quả đầy đủ: {output_path}")
        
        # Lưu riêng các anomalies
        anomalies_path = output_path.replace('.json', '_anomalies_only.json')
        anomalies = result_df[result_df['is_anomaly'] == 1].copy()
        anomalies = anomalies.sort_values('anomaly_score')
        anomalies.to_json(anomalies_path, orient='records', indent=2, force_ascii=False)
        print(f"✓ Đã lưu giao dịch bất thường: {anomalies_path}")
        
        return anomalies


def main():
    """
    Hàm chính để chạy phát hiện anomaly
    """
    print("="*70)
    print("🌲 ISOLATION FOREST - PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG 🌲")
    print("="*70)
    
    # Đường dẫn file dữ liệu
    data_path = 'dataset1.0/transactions.json'
    
    # Khởi tạo detector
    # contamination=0.1 nghĩa là dự kiến 10% dữ liệu là anomaly
    # Bạn có thể điều chỉnh giá trị này (0.05 = 5%, 0.15 = 15%, ...)
    detector = TransactionAnomalyDetector(contamination=0.1)
    
    # Tải dữ liệu
    df = detector.load_data(data_path)
    
    # Phát hiện anomalies
    result_df = detector.detect_anomalies(df)
    
    # Lấy top 20 giao dịch bất thường nhất
    print("\n" + "="*70)
    print("🔴 TOP 20 GIAO DỊCH BẤT THƯỜNG NHẤT")
    print("="*70)
    
    top_anomalies = detector.get_top_anomalies(result_df, top_n=20)
    
    for idx, (i, row) in enumerate(top_anomalies.iterrows(), 1):
        print(f"\n--- #{idx} - Anomaly Score: {row['anomaly_score']:.4f} ---")
        # In ra các thông tin quan trọng của giao dịch
        for col in df.columns:
            if col not in ['anomaly_score', 'is_anomaly', 'anomaly_rank']:
                print(f"  {col}: {row[col]}")
    
    # Trực quan hóa kết quả
    print("\n" + "="*70)
    detector.visualize_anomalies(result_df)
    
    # Lưu kết quả
    print("\n" + "="*70)
    print("💾 ĐANG LƯU KẾT QUẢ...")
    print("="*70)
    anomalies = detector.save_results(result_df)
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    print(f"📁 File kết quả:")
    print(f"  - anomaly_results.json (toàn bộ giao dịch)")
    print(f"  - anomaly_results_anomalies_only.json (chỉ giao dịch bất thường)")
    print(f"  - anomaly_visualization.png (biểu đồ trực quan)")
    print("="*70)


if __name__ == "__main__":
    main()
