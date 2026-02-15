# Isolation Forest - Thuật toán Phát hiện Anomaly

## Giới thiệu

**Isolation Forest** (Rừng Cô lập) là một thuật toán machine learning được thiết kế đặc biệt để phát hiện **outliers** (điểm bất thường) trong dữ liệu. Thuật toán này được phát triển bởi Fei Tony Liu và cộng sự vào năm 2008.

### Ý tưởng cốt lõi

> **"Những điểm bất thường dễ dàng bị cô lập hơn những điểm bình thường"**

Thay vì mô hình hóa các điểm bình thường (như các thuật toán phát hiện anomaly truyền thống), Isolation Forest **trực tiếp cô lập các điểm bất thường**.

---

## Nguyên lý hoạt động

### 1. Tại sao Anomalies dễ bị cô lập?

Hãy tưởng tượng bạn có một tập dữ liệu về **chiều cao và cân nặng** của người:

```
Dữ liệu bình thường:
- Người 1: Cao 1.7m, nặng 65kg
- Người 2: Cao 1.65m, nặng 60kg
- Người 3: Cao 1.75m, nặng 70kg
...

Dữ liệu bất thường:
- Người X: Cao 2.5m, nặng 150kg (Quá cao và quá nặng)
```

Khi bạn vẽ một đường phân chia ngẫu nhiên:
- Để tách **Người X** ra khỏi nhóm: chỉ cần **1-2 lần cắt**
- Để tách **Người 1** ra khỏi nhóm: cần **nhiều lần cắt hơn** (vì nằm giữa đám đông)

**Điểm bất thường = Cần ít lần cắt hơn để cô lập**

---

## Cách Isolation Forest hoạt động

### Bước 1: Xây dựng Isolation Trees (Cây Cô lập)

1. **Chọn ngẫu nhiên** một đặc trưng (feature) từ dữ liệu
2. **Chọn ngẫu nhiên** một giá trị phân chia giữa min và max của feature đó
3. **Chia dữ liệu** thành 2 nhóm dựa trên giá trị phân chia
4. **Lặp lại** cho đến khi mỗi điểm dữ liệu bị cô lập hoặc đạt độ sâu tối đa

```
Ví dụ Isolation Tree:

                [Tất cả dữ liệu]
                       |
        Số tiền < $5000? (Phân chia ngẫu nhiên)
              /                \
           Có                  Không
           |                      |
    Địa điểm = VN?         Thiết bị = Emulator?
      /        \              /          \
    Có        Không         Có         Không
    |           |           |             |
 [Bình thường][...]    [ANOMALY!]      [...]
```

### Bước 2: Xây dựng Forest (Rừng)

- Tạo **nhiều cây** (thường là 100-200 cây)
- Mỗi cây được xây dựng với **mẫu ngẫu nhiên** từ dữ liệu
- Mỗi cây có **cách phân chia ngẫu nhiên** khác nhau

### Bước 3: Tính Anomaly Score

Đối với mỗi điểm dữ liệu:

1. **Đo độ sâu cô lập** (path length) trên mỗi cây
2. **Tính trung bình** độ sâu trên tất cả các cây
3. **Chuẩn hóa** để có điểm anomaly score

```python
# Công thức tính Anomaly Score
anomaly_score = 2^(-average_path_length / c(n))

# Trong đó:
# - average_path_length: độ sâu trung bình để cô lập điểm
# - c(n): hằng số chuẩn hóa dựa trên số lượng mẫu
```

### Giải thích Anomaly Score

| Score | Ý nghĩa |
|-------|---------|
| **Score ≈ 1** | Rất bất thường (anomaly) |
| **Score ≈ 0.5** | Không chắc chắn (cần xem xét) |
| **Score < 0.5** | Bình thường (normal) |

> **Lưu ý**: Trong scikit-learn, score càng **âm** càng bất thường!

---

## Ưu điểm của Isolation Forest

### 1. Hiệu quả cao
- Không cần huấn luyện trên toàn bộ dữ liệu
- Chỉ cần một mẫu nhỏ ngẫu nhiên

### 2. Nhanh
- Độ phức tạp: **O(n log n)** - rất nhanh!
- Có thể xử lý dataset lớn

### 3. Không cần label
- **Unsupervised learning** - không cần dữ liệu được gán nhãn
- Tự động phát hiện anomaly

### 4. Xử lý tốt với dữ liệu nhiều chiều
- Hoạt động tốt với nhiều features
- Không bị ảnh hưởng bởi curse of dimensionality

### 5. Ít bị ảnh hưởng bởi outliers
- Vì chỉ tập trung vào việc cô lập, không mô hình hóa phân phối

---

## Nhược điểm

### 1. Khó xác định ngưỡng
- Cần thiết lập `contamination` (tỷ lệ anomaly) trước
- Nếu không biết tỷ lệ, có thể cho kết quả không chính xác

### 2. Kém hiệu quả với anomalies cục bộ
- Nếu anomalies tạo thành nhóm riêng, có thể không phát hiện được

### 3. Kết quả không ổn định
- Do tính ngẫu nhiên, mỗi lần chạy có thể cho kết quả khác nhau
- Cần đặt `random_state` để có kết quả nhất quán

---

## Các tham số quan trọng

### 1. `n_estimators` (Số lượng cây)
```python
n_estimators=100  # Mặc định
# Càng nhiều cây → kết quả càng ổn định
# Nhưng tốn thời gian hơn
```

### 2. `contamination` (Tỷ lệ anomaly)
```python
contamination=0.1  # 10% dữ liệu là anomaly
contamination=0.05 # 5% dữ liệu là anomaly
contamination='auto'  # Tự động xác định
```

### 3. `max_samples` (Số mẫu cho mỗi cây)
```python
max_samples='auto'  # Mặc định: min(256, n_samples)
max_samples=512     # Tùy chỉnh
```

### 4. `random_state` (Seed ngẫu nhiên)
```python
random_state=42  # Đảm bảo kết quả nhất quán
```

---

## Ví dụ thực tế: Phát hiện Giao dịch Gian lận

### Tình huống
Bạn có 5000 giao dịch ngân hàng và muốn tìm giao dịch gian lận.

### Các đặc trưng (features)
1. **Transaction amount** - Số tiền giao dịch
2. **Time of day** - Thời gian trong ngày
3. **Location** - Địa điểm
4. **Device** - Thiết bị sử dụng

### Dữ liệu bình thường
```
- Số tiền: $50 - $500
- Thời gian: 8am - 10pm
- Địa điểm: Việt Nam
- Thiết bị: iPhone, Samsung
```

### Dữ liệu bất thường (Anomalies)
```
- Số tiền: $15,000
- Thời gian: 3am
- Địa điểm: Quốc tế (Tokyo, Paris)
- Thiết bị: Android Emulator
```

### Kết quả từ Isolation Forest

Top 3 giao dịch bất thường nhất:

```
1. TXN_1002 - Score: -0.615
   - $7,270 - Crypto exchange funding
   - Singapore - Orchard Road
   - Android Emulator (CẢNH BÁO)
   
2. TXN_1032 - Score: -0.615
   - $13,795 - Offshore service fee
   - Tokyo - Shibuya
   - Google Pixel 8
   
3. TXN_1103 - Score: -0.608
   - $14,580 - Gaming wallet top-up
   - Tokyo - Shibuya
   - Android Emulator (CẢNH BÁO)
```

### Tại sao những giao dịch này bất thường?

1. **Số tiền rất cao** so với trung bình
2. **Loại giao dịch nguy hiểm**: Crypto, Gaming, Offshore
3. **Địa điểm quốc tế** thay vì trong nước
4. **Thiết bị đáng ngờ**: Android Emulator (công cụ của hacker!)

---

## Code Implementation

### Cài đặt thư viện
```bash
pip install scikit-learn pandas numpy
```

### Code cơ bản

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# 1. Chuẩn bị dữ liệu
X = np.array([
    [100, 1, 0],    # Giao dịch bình thường
    [150, 2, 0],    # Giao dịch bình thường
    [120, 1, 0],    # Giao dịch bình thường
    [15000, 5, 1],  # GD BẤT THƯỜNG: Số tiền cao, địa điểm nước ngoài
    [130, 2, 0],    # Giao dịch bình thường
])

# 2. Tạo và huấn luyện model
model = IsolationForest(
    contamination=0.2,  # 20% dữ liệu là anomaly
    random_state=42
)
model.fit(X)

# 3. Dự đoán anomalies
predictions = model.predict(X)
# Kết quả: [1, 1, 1, -1, 1]
# 1 = normal, -1 = anomaly

# 4. Tính anomaly score
scores = model.score_samples(X)
# Score càng âm = càng bất thường
print(scores)
```

### Code nâng cao với dữ liệu thực tế

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Load dữ liệu
df = pd.read_json('transactions.json')

# 2. Trích xuất features
features = df[['Transaction amount', 'hour', 'location_code']]

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 4. Huấn luyện Isolation Forest
clf = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # 10% anomalies
    random_state=42
)
clf.fit(X_scaled)

# 5. Phát hiện anomalies
df['anomaly_score'] = clf.score_samples(X_scaled)
df['is_anomaly'] = clf.predict(X_scaled)

# 6. Lấy top anomalies
anomalies = df[df['is_anomaly'] == -1].sort_values('anomaly_score')
print(f"Tìm thấy {len(anomalies)} giao dịch bất thường!")
```

---

## So sánh với các thuật toán khác

| Thuật toán | Ưu điểm | Nhược điểm | Tốc độ |
|------------|---------|------------|--------|
| **Isolation Forest** | Nhanh, hiệu quả, dễ sử dụng | Cần biết tỷ lệ contamination | Rất nhanh |
| **LOF** (Local Outlier Factor) | Tốt với anomalies cục bộ | Chậm với dữ liệu lớn | Chậm |
| **One-Class SVM** | Tốt với dữ liệu phi tuyến | Khó tune parameters | Rất chậm |
| **DBSCAN** | Tự động tìm clusters | Khó thiết lập parameters | Trung bình |
| **Statistical Methods** | Dễ hiểu, giải thích được | Giả định phân phối chuẩn | Nhanh |

---

## Best Practices (Thực hành tốt nhất)

### 1. Chuẩn hóa dữ liệu
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Thử nghiệm nhiều giá trị contamination
```python
for contamination in [0.05, 0.1, 0.15, 0.2]:
    clf = IsolationForest(contamination=contamination)
    clf.fit(X)
    # Đánh giá kết quả...
```

### 3. Sử dụng cross-validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, cv=5)
```

### 4. Kết hợp với domain knowledge
- Không chỉ dựa vào anomaly score
- Kết hợp với kiến thức chuyên môn để xác nhận

### 5. Visualize kết quả
```python
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=predictions)
plt.title('Isolation Forest Results')
plt.show()
```

---

## Tài liệu tham khảo

1. **Paper gốc**: Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.

2. **Scikit-learn Documentation**: 
   - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

3. **Tutorials**:
   - https://towardsdatascience.com/isolation-forest-algorithm

---

## Khi nào nên dùng Isolation Forest?

### Nên dùng khi:
- Cần phát hiện anomalies **nhanh chóng**
- Dữ liệu có **nhiều features**
- Không biết chính xác anomaly trông như thế nào
- Có **dataset lớn**
- Cần thuật toán **dễ implement**

### Không nên dùng khi:
- Anomalies tạo thành **nhóm riêng** (dùng DBSCAN thay thế)
- Cần hiểu **tại sao** một điểm là anomaly (dùng statistical methods)
- Dữ liệu có **cấu trúc phức tạp** (dùng deep learning)
- Dataset **rất nhỏ** (< 100 samples)

---

## Kết luận

**Isolation Forest** là một thuật toán mạnh mẽ, nhanh chóng và hiệu quả để phát hiện anomalies. Với nguyên lý **"cô lập thay vì mô hình hóa"**, nó đã trở thành một trong những phương pháp phổ biến nhất trong anomaly detection.

### Key Takeaways:
1. **Anomalies dễ bị cô lập hơn** điểm bình thường
2. **Nhanh và hiệu quả** với O(n log n)
3. **Unsupervised** - không cần data được gán nhãn
4. **Dễ sử dụng** với scikit-learn
5. **Cần tune** tham số `contamination` cẩn thận

---

**Happy Anomaly Detection!**
