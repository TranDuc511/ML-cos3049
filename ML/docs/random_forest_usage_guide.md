# Random Forest Classification - Hướng Dẫn Sử Dụng

## Mục Lục
1. [Cài Đặt](#cài-đặt)
2. [Sử Dụng Cơ Bản](#sử-dụng-cơ-bản)
3. [Các Tính Năng Nâng Cao](#các-tính-năng-nâng-cao)
4. [Ví Dụ Thực Tế](#ví-dụ-thực-tế)
5. [Tối Ưu Hóa Hiệu Suất](#tối-ưu-hóa-hiệu-suất)
6. [Xử Lý Lỗi Thường Gặp](#xử-lý-lỗi-thường-gặp)

---

## Cài Đặt

### Yêu Cầu Hệ Thống
```bash
Python >= 3.7
numpy >= 1.19.0
pandas >= 1.1.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
joblib >= 1.0.0
```

### Cài Đặt Thư Viện
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

---

## Sử Dụng Cơ Bản

### 1. Import Module
```python
from random_forest import RandomForestClassificationModel
import pandas as pd
import numpy as np
```

### 2. Chuẩn Bị Dữ Liệu
```python
# Đọc dữ liệu từ file CSV
data = pd.read_csv('your_data.csv')

# Tách features và target
X = data.drop('target_column', axis=1)
y = data['target_column']

# Chia tập train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Tạo và Huấn Luyện Mô Hình
```python
# Khởi tạo mô hình với tham số mặc định
model = RandomForestClassificationModel(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Huấn luyện mô hình
model.fit(X_train, y_train)
```

### 4. Dự Đoán
```python
# Dự đoán nhãn
predictions = model.predict(X_test)

# Dự đoán xác suất
probabilities = model.predict_proba(X_test)
print(f"Xác suất cho mẫu đầu tiên: {probabilities[0]}")
```

### 5. Đánh Giá Mô Hình
```python
# Đánh giá toàn diện
metrics = model.evaluate(X_test, y_test)

print("Kết quả đánh giá:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

**Output mẫu:**
```
Kết quả đánh giá:
  accuracy: 0.8750
  precision: 0.8723
  recall: 0.8750
  f1_score: 0.8731
  roc_auc: 0.9234
```

---

## Các Tính Năng Nâng Cao

### 1. Phân Tích Tầm Quan Trọng Đặc Trưng

#### Lấy Danh Sách Đặc Trưng Quan Trọng
```python
# Lấy top 10 đặc trưng quan trọng nhất
importance_df = model.get_feature_importance(top_n=10)
print(importance_df)
```

**Output:**
```
           feature  importance
0         feature_5    0.234521
1        feature_12    0.187634
2         feature_3    0.145892
3         feature_8    0.098234
...
```

#### Vẽ Biểu Đồ Tầm Quan Trọng
```python
# Vẽ biểu đồ cho top 15 đặc trưng
model.plot_feature_importance(top_n=15, figsize=(12, 8))
```

### 2. Ma Trận Nhầm Lẫn (Confusion Matrix)
```python
# Vẽ confusion matrix
model.plot_confusion_matrix(X_test, y_test, figsize=(10, 8))
```

### 3. Cross-Validation (Kiểm Định Chéo)
```python
# Thực hiện 5-fold cross-validation
cv_results = model.cross_validate(X_train, y_train, cv=5)

print(f"Điểm trung bình: {cv_results['mean_score']:.4f}")
print(f"Độ lệch chuẩn: {cv_results['std_score']:.4f}")
print(f"Điểm từng fold: {cv_results['scores']}")
```

**Output mẫu:**
```
Điểm trung bình: 0.8623
Độ lệch chuẩn: 0.0234
Điểm từng fold: [0.8456 0.8723 0.8891 0.8512 0.8534]
```

### 4. Tối Ưu Hóa Tham Số (Hyperparameter Tuning)

#### Tùy Chỉnh Lưới Tham Số
```python
# Định nghĩa lưới tham số cần tìm kiếm
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Thực hiện tìm kiếm
results = model.hyperparameter_tuning(
    X_train, y_train,
    param_grid=param_grid,
    cv=5
)

print(f"Tham số tốt nhất: {results['best_params']}")
print(f"Điểm tốt nhất: {results['best_score']:.4f}")
```

#### Sử Dụng Lưới Mặc Định
```python
# Sử dụng lưới tham số mặc định
results = model.hyperparameter_tuning(X_train, y_train, cv=5)
```

### 5. Lưu và Tải Mô Hình

#### Lưu Mô Hình
```python
# Lưu mô hình đã huấn luyện
model.save_model('models/random_forest_model.pkl')
```

#### Tải Mô Hình
```python
# Tải mô hình đã lưu
loaded_model = RandomForestClassificationModel.load_model(
    'models/random_forest_model.pkl'
)

# Sử dụng mô hình đã tải
predictions = loaded_model.predict(X_test)
```

---

## Ví Dụ Thực Tế

### Ví Dụ 1: Phân Loại Khách Hàng

```python
import pandas as pd
from random_forest import RandomForestClassificationModel
from sklearn.model_selection import train_test_split

# Đọc dữ liệu khách hàng
customer_data = pd.read_csv('customer_data.csv')

# Chuẩn bị dữ liệu
X = customer_data[['age', 'income', 'purchase_frequency', 'avg_spend']]
y = customer_data['customer_segment']  # 0: Bronze, 1: Silver, 2: Gold

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Tạo và huấn luyện mô hình
model = RandomForestClassificationModel(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train, feature_names=X.columns.tolist())

# Đánh giá
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Xem đặc trưng quan trọng
importance = model.get_feature_importance()
print("\nĐặc trưng quan trọng:")
print(importance)

# Dự đoán khách hàng mới
new_customer = [[35, 75000, 5, 250]]  # age, income, frequency, avg_spend
segment = model.predict(new_customer)
proba = model.predict_proba(new_customer)

print(f"\nKhách hàng thuộc phân khúc: {segment[0]}")
print(f"Xác suất: Bronze={proba[0][0]:.2%}, Silver={proba[0][1]:.2%}, Gold={proba[0][2]:.2%}")
```

### Ví Dụ 2: Phát Hiện Gian Lận

```python
# Đọc dữ liệu giao dịch
transactions = pd.read_csv('transactions.csv')

# Chuẩn bị features
X = transactions[[
    'amount', 'merchant_category', 'time_of_day',
    'location', 'card_present', 'online_transaction'
]]
y = transactions['is_fraud']  # 0: Hợp lệ, 1: Gian lận

# One-hot encoding cho categorical features
X = pd.get_dummies(X, columns=['merchant_category', 'location'])

# Chia dữ liệu (lưu ý: dữ liệu không cân bằng)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Tạo mô hình với class_weight để xử lý dữ liệu không cân bằng
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

model = RandomForestClassificationModel(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42
)

# Note: Để sử dụng class_weight, cần truy cập trực tiếp model.model
model.model.set_params(class_weight='balanced')

model.fit(X_train, y_train)

# Đánh giá với chú trọng vào Recall (phát hiện gian lận)
metrics = model.evaluate(X_test, y_test)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")

# Vẽ confusion matrix
model.plot_confusion_matrix(X_test, y_test)
```

### Ví Dụ 3: Phân Loại Email Spam

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Đọc dữ liệu email
emails = pd.read_csv('emails.csv')

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(emails['email_text'])
y = emails['is_spam']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tạo mô hình
model = RandomForestClassificationModel(
    n_estimators=150,
    max_depth=25,
    random_state=42
)

model.fit(X_train.toarray(), y_train)

# Đánh giá
metrics = model.evaluate(X_test.toarray(), y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Dự đoán email mới
new_email = ["Congratulations! You won a prize. Click here to claim."]
new_email_vec = vectorizer.transform(new_email)
prediction = model.predict(new_email_vec.toarray())
print(f"Email này là: {'SPAM' if prediction[0] == 1 else 'HAM (Hợp lệ)'}")
```

---

## Tối Ưu Hóa Hiệu Suất

### 1. Tăng Tốc Huấn Luyện

#### Sử Dụng Parallel Processing
```python
# Sử dụng tất cả CPU cores
model = RandomForestClassificationModel(
    n_estimators=200,
    n_jobs=-1  # Sử dụng tất cả cores
)
```

#### Giảm Số Lượng Cây (Nếu Cần)
```python
# Trade-off giữa accuracy và speed
model = RandomForestClassificationModel(
    n_estimators=50,  # Giảm từ 100 xuống 50
    random_state=42
)
```

### 2. Giảm Bộ Nhớ

#### Giới Hạn Độ Sâu Cây
```python
model = RandomForestClassificationModel(
    n_estimators=100,
    max_depth=15,  # Giới hạn độ sâu
    min_samples_split=10,
    min_samples_leaf=5
)
```

### 3. Cải Thiện Accuracy

#### Tăng Số Cây
```python
model = RandomForestClassificationModel(
    n_estimators=500,  # Tăng số cây
    random_state=42
)
```

#### Feature Engineering
```python
# Tạo features mới từ features hiện có
data['age_income_ratio'] = data['age'] / data['income']
data['frequency_spend_product'] = data['purchase_frequency'] * data['avg_spend']
```

---

## Xử Lý Lỗi Thường Gặp

### Lỗi 1: Model Must Be Fitted First
```python
# ❌ Sai
model = RandomForestClassificationModel()
predictions = model.predict(X_test)  # Lỗi!

# ✅ Đúng
model = RandomForestClassificationModel()
model.fit(X_train, y_train)  # Huấn luyện trước
predictions = model.predict(X_test)  # OK
```

### Lỗi 2: Shape Mismatch
```python
# ❌ Sai - Số features không khớp
model.fit(X_train, y_train)  # X_train có 10 features
predictions = model.predict(X_test_wrong)  # X_test có 8 features

# ✅ Đúng - Đảm bảo features khớp
assert X_train.shape[1] == X_test.shape[1]
predictions = model.predict(X_test)
```

### Lỗi 3: Dữ Liệu Chứa NaN
```python
# ❌ Sai
model.fit(X_train, y_train)  # X_train có NaN values

# ✅ Đúng - Xử lý missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

model.fit(X_train_imputed, y_train)
```

### Lỗi 4: Memory Error
```python
# ❌ Sai - Quá nhiều cây và dữ liệu lớn
model = RandomForestClassificationModel(n_estimators=1000)
model.fit(huge_dataset, y)  # Out of memory!

# ✅ Đúng - Giảm số cây hoặc sample dữ liệu
# Option 1: Giảm số cây
model = RandomForestClassificationModel(n_estimators=100)

# Option 2: Sample dữ liệu
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
for train_idx, _ in splitter.split(huge_dataset, y):
    X_sample = huge_dataset[train_idx]
    y_sample = y[train_idx]
    model.fit(X_sample, y_sample)
```

---

## Best Practices

### 1. Luôn Chia Tập Train/Test
```python
# Sử dụng stratify để giữ tỉ lệ các class
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 2. Sử Dụng Cross-Validation
```python
# Đánh giá mô hình chính xác hơn
cv_results = model.cross_validate(X_train, y_train, cv=5)
```

### 3. Theo Dõi Feature Importance
```python
# Xác định features quan trọng để tối ưu hóa
importance = model.get_feature_importance()
# Loại bỏ features không quan trọng nếu cần
```

### 4. Lưu Mô Hình Sau Khi Huấn Luyện
```python
# Tránh phải huấn luyện lại
model.save_model(f'model_{datetime.now().strftime("%Y%m%d")}.pkl')
```

### 5. Xử Lý Imbalanced Data
```python
# Sử dụng class_weight hoặc SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model.fit(X_resampled, y_resampled)
```

---

## Tài Liệu Tham Khảo

- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Random Forest Overview](random_forest_overview.md)
- [Source Code](../src/random_forest.py)

---

**Cập nhật lần cuối**: 2026-02-08
