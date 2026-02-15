# ALGO - Isolation Forest Anomaly Detection

Dự án phát hiện giao dịch bất thường bằng thuật toán Isolation Forest.

## Cấu trúc thư mục

```
ALGO/
├── data/                   # Dữ liệu gốc
│   ├── transactions.json   # Dữ liệu giao dịch
│   └── customers_fixed.json # Dữ liệu khách hàng
│
├── src/                    # Source code
│   └── isolation_forest_anomaly_detection.py  # Script phát hiện anomaly
│
├── results/                # Kết quả phân tích
│   ├── anomaly_results.json                   # Toàn bộ kết quả
│   └── anomaly_results_anomalies_only.json    # Chỉ giao dịch bất thường
│
├── visualizations/         # Biểu đồ và hình ảnh
│   └── anomaly_visualization.png              # Biểu đồ phân tích
│
├── docs/                   # Tài liệu
│   └── ISOLATION_FOREST_EXPLAINED.md          # Giải thích thuật toán
│
└── README.md              # File này
```

## Giới thiệu

Dự án sử dụng **Isolation Forest** để tự động phát hiện các giao dịch bất thường (anomalies) trong dữ liệu giao dịch ngân hàng. Thuật toán hoạt động dựa trên nguyên lý: "Những điểm bất thường dễ dàng bị cô lập hơn những điểm bình thường".

## Yêu cầu

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

## Cách sử dụng

### 1. Chạy phân tích

```bash
cd src
python isolation_forest_anomaly_detection.py
```

### 2. Xem kết quả

- **Kết quả chi tiết**: `results/anomaly_results.json`
- **Chỉ anomalies**: `results/anomaly_results_anomalies_only.json`
- **Biểu đồ trực quan**: `visualizations/anomaly_visualization.png`

### 3. Tìm hiểu thuật toán

Đọc tài liệu chi tiết tại: `docs/ISOLATION_FOREST_EXPLAINED.md`

## Kết quả

Script đã phát hiện **50 giao dịch bất thường** từ tổng số 5000 giao dịch (10%).

### Top 3 giao dịch bất thường nhất:

1. **TXN_1002** (Score: -0.615)
   - Số tiền: $7,270
   - Loại: Crypto exchange funding
   - Địa điểm: Singapore
   - Thiết bị: Android Emulator (Cảnh báo!)

2. **TXN_1032** (Score: -0.615)
   - Số tiền: $13,795
   - Loại: Offshore service fee
   - Địa điểm: Tokyo

3. **TXN_1103** (Score: -0.608)
   - Số tiền: $14,580
   - Loại: Gaming wallet top-up
   - Địa điểm: Tokyo
   - Thiết bị: Android Emulator (Cảnh báo!)

## Đặc điểm giao dịch bất thường

- Số tiền cao bất thường (>$7,000)
- Loại giao dịch nguy hiểm: Crypto, Gaming, P2P loan, Offshore
- Địa điểm quốc tế: Tokyo, Singapore, Paris, New York
- Thiết bị đáng ngờ: Android Emulator

## Tùy chỉnh

Để điều chỉnh tỷ lệ phát hiện anomaly, sửa tham số `contamination` trong file `src/isolation_forest_anomaly_detection.py`:

```python
detector = TransactionAnomalyDetector(contamination=0.1)  # 10% anomalies
# Thay đổi thành 0.05 (5%), 0.15 (15%), v.v.
```

## Tham khảo

- Paper gốc: Liu, Fei Tony, et al. "Isolation forest." 2008
- Scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

## License

MIT License
