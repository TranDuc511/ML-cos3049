# Random Forest Classification - Ví Dụ Chi Tiết

## Mục Lục
1. [Ví Dụ Cơ Bản với Dữ Liệu Synthetic](#ví-dụ-cơ-bản-với-dữ-liệu-synthetic)
2. [Dự Đoán Chất Lượng Rượu](#dự-đoán-chất-lượng-rượu)
3. [Phân Loại Bệnh Tiểu Đường](#phân-loại-bệnh-tiểu-đường)
4. [Dự Đoán Sinh Viên Có Tốt Nghiệp Đúng Hạn](#dự-đoán-sinh-viên-có-tốt-nghiệp-đúng-hạn)
5. [Phân Tích A/B Testing](#phân-tích-ab-testing)

---

## Ví Dụ Cơ Bản với Dữ Liệu Synthetic

### Mục Tiêu
Minh họa cách sử dụng cơ bản của Random Forest với dữ liệu tạo sẵn.

### Code Hoàn Chỉnh

```python
from random_forest import RandomForestClassificationModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Tạo dữ liệu synthetic
print("Tạo dữ liệu...")
X, y = make_classification(
    n_samples=1000,        # 1000 mẫu
    n_features=20,         # 20 đặc trưng
    n_informative=15,      # 15 đặc trưng có ích
    n_redundant=5,         # 5 đặc trưng dư thừa
    n_classes=2,           # Phân loại nhị phân
    random_state=42
)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Kích thước tập train: {X_train.shape}")
print(f"Kích thước tập test: {X_test.shape}")

# Tạo và huấn luyện mô hình
print("\nHuấn luyện mô hình...")
model = RandomForestClassificationModel(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Đánh giá
print("\nĐánh giá mô hình...")
metrics = model.evaluate(X_test, y_test)

print("\nKết quả:")
print(f"  Accuracy:  {metrics['accuracy']:.2%}")
print(f"  Precision: {metrics['precision']:.2%}")
print(f"  Recall:    {metrics['recall']:.2%}")
print(f"  F1-Score:  {metrics['f1_score']:.2%}")
if 'roc_auc' in metrics:
    print(f"  ROC AUC:   {metrics['roc_auc']:.2%}")

# Vẽ feature importance
model.plot_feature_importance(top_n=10)

# Cross-validation
print("\nThực hiện Cross-Validation...")
cv_results = model.cross_validate(X_train, y_train, cv=5)
print(f"CV Mean Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
```

### Kết Quả Mẫu
```
Kích thước tập train: (800, 20)
Kích thước tập test: (200, 20)

Huấn luyện mô hình...
Model trained successfully with 100 trees

Đánh giá mô hình...

Kết quả:
  Accuracy:  87.50%
  Precision: 87.23%
  Recall:    87.50%
  F1-Score:  87.31%
  ROC AUC:   92.34%

Thực hiện Cross-Validation...
CV Mean Score: 0.8625 (+/- 0.0234)
```

---

## Dự Đoán Chất Lượng Rượu

### Mục Tiêu
Phân loại rượu vang đỏ thành "chất lượng tốt" hoặc "chất lượng trung bình/kém" dựa trên các đặc tính hóa học.

### Dữ Liệu
```python
import pandas as pd
import numpy as np
from random_forest import RandomForestClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Tải dữ liệu Wine Quality
# Dataset: https://archive.ics.uci.edu/ml/datasets/wine+quality
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_data = pd.read_csv(url, sep=';')

# Xem thông tin dữ liệu
print("Thông tin dữ liệu:")
print(wine_data.info())
print("\nThống kê mô tả:")
print(wine_data.describe())

# Kiểm tra missing values
print(f"\nMissing values: {wine_data.isnull().sum().sum()}")
```

### Chuẩn Bị Dữ Liệu

```python
# Chuyển đổi bài toán thành phân loại nhị phân
# Quality >= 7: Tốt (1), Quality < 7: Trung bình/Kém (0)
wine_data['quality_binary'] = (wine_data['quality'] >= 7).astype(int)

print("\nPhân bố nhãn:")
print(wine_data['quality_binary'].value_counts())
print(f"Tỉ lệ: {wine_data['quality_binary'].value_counts(normalize=True)}")

# Tách features và target
X = wine_data.drop(['quality', 'quality_binary'], axis=1)
y = wine_data['quality_binary']

# Chuẩn hóa dữ liệu (không bắt buộc cho Random Forest nhưng tốt hơn)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập train/test với stratify để giữ tỉ lệ
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nKích thước train: {X_train.shape}")
print(f"Kích thước test: {X_test.shape}")
```

### Huấn Luyện Mô Hình

```python
# Tạo mô hình
model = RandomForestClassificationModel(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Huấn luyện
print("\nHuấn luyện mô hình...")
model.fit(X_train, y_train, feature_names=X.columns.tolist())

# Đánh giá
print("\nĐánh giá trên tập test:")
metrics = model.evaluate(X_test, y_test)
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# Cross-validation
print("\nCross-Validation (5-fold):")
cv_results = model.cross_validate(X_train, y_train, cv=5)
print(f"  Mean: {cv_results['mean_score']:.4f}")
print(f"  Std:  {cv_results['std_score']:.4f}")
```

### Phân Tích Feature Importance

```python
# Xem đặc trưng quan trọng
print("\nĐặc trưng quan trọng nhất:")
importance_df = model.get_feature_importance()
print(importance_df)

# Vẽ biểu đồ
model.plot_feature_importance(top_n=11, figsize=(10, 8))

# Vẽ confusion matrix
model.plot_confusion_matrix(X_test, y_test, figsize=(8, 6))
```

### Dự Đoán Mẫu Mới

```python
# Tạo mẫu rượu mới (chuẩn hóa giống như dữ liệu huấn luyện)
new_wine = pd.DataFrame({
    'fixed acidity': [7.4],
    'volatile acidity': [0.70],
    'citric acid': [0.00],
    'residual sugar': [1.9],
    'chlorides': [0.076],
    'free sulfur dioxide': [11.0],
    'total sulfur dioxide': [34.0],
    'density': [0.9978],
    'pH': [3.51],
    'sulphates': [0.56],
    'alcohol': [9.4]
})

# Chuẩn hóa
new_wine_scaled = scaler.transform(new_wine)

# Dự đoán
prediction = model.predict(new_wine_scaled)
probability = model.predict_proba(new_wine_scaled)

print("\n" + "="*50)
print("DỰ ĐOÁN MẪU RƯỢU MỚI")
print("="*50)
print(f"Chất lượng: {'TỐT' if prediction[0] == 1 else 'TRUNG BÌNH/KÉM'}")
print(f"Xác suất Trung bình/Kém: {probability[0][0]:.2%}")
print(f"Xác suất Tốt: {probability[0][1]:.2%}")
```

### Kết Quả Mẫu

```
Đặc trưng quan trọng nhất:
              feature  importance
0             alcohol    0.145623
1           sulphates    0.123456
2       volatile acidity    0.112345
3         total sulfur dioxide    0.098765
4              density    0.087654

DỰ ĐOÁN MẪU RƯỢU MỚI
==================================================
Chất lượng: TRUNG BÌNH/KÉM
Xác suất Trung bình/Kém: 76.34%
Xác suất Tốt: 23.66%
```

---

## Phân Loại Bệnh Tiểu Đường

### Mục Tiêu
Dự đoán liệu một bệnh nhân có bị tiểu đường hay không dựa trên các yếu tố sức khỏe.

### Dữ Liệu và Chuẩn Bị

```python
import pandas as pd
from random_forest import RandomForestClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Tải dữ liệu Pima Indians Diabetes
# Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
diabetes_data = pd.read_csv('diabetes.csv')

print("Thông tin dữ liệu:")
print(diabetes_data.head())
print(f"\nKích thước: {diabetes_data.shape}")
print(f"\nPhân bố nhãn:\n{diabetes_data['Outcome'].value_counts()}")

# Kiểm tra missing values (được mã hóa là 0 trong dataset này)
print("\nKiểm tra giá trị 0 (có thể là missing):")
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    zero_count = (diabetes_data[col] == 0).sum()
    print(f"  {col}: {zero_count} giá trị 0")

# Thay thế 0 bằng median (hoặc mean)
for col in zero_cols:
    diabetes_data[col] = diabetes_data[col].replace(0, diabetes_data[col].median())

# Tách features và target
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
```

### Huấn Luyện và Đánh Giá

```python
# Tạo mô hình với class_weight='balanced' cho imbalanced data
model = RandomForestClassificationModel(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

# Thiết lập class_weight
model.model.set_params(class_weight='balanced')

# Huấn luyện
print("\nHuấn luyện mô hình...")
model.fit(X_train, y_train, feature_names=X.columns.tolist())

# Đánh giá
print("\nKết quả trên tập test:")
metrics = model.evaluate(X_test, y_test)
for metric, value in metrics.items():
    print(f"  {metric.capitalize():12s}: {value:.4f}")

# Confusion Matrix
model.plot_confusion_matrix(X_test, y_test)

# Feature Importance
print("\nĐặc trưng quan trọng:")
importance_df = model.get_feature_importance()
print(importance_df)
model.plot_feature_importance(top_n=8)
```

### Tối Ưu Hóa Hyperparameters

```python
# Tìm kiếm tham số tối ưu
print("\nTối ưu hóa hyperparameters...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6]
}

tuning_results = model.hyperparameter_tuning(
    X_train, y_train,
    param_grid=param_grid,
    cv=5
)

print(f"\nTham số tốt nhất: {tuning_results['best_params']}")
print(f"Điểm tốt nhất: {tuning_results['best_score']:.4f}")

# Đánh giá lại với mô hình đã tối ưu
final_metrics = model.evaluate(X_test, y_test)
print("\nKết quả sau khi tối ưu:")
for metric, value in final_metrics.items():
    print(f"  {metric.capitalize():12s}: {value:.4f}")
```

### Dự Đoán Bệnh Nhân Mới

```python
# Thông tin bệnh nhân mới
new_patient = pd.DataFrame({
    'Pregnancies': [6],
    'Glucose': [148],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [125],
    'BMI': [33.6],
    'DiabetesPedigreeFunction': [0.627],
    'Age': [50]
})

# Chuẩn hóa
new_patient_scaled = scaler.transform(new_patient)

# Dự đoán
prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)

print("\n" + "="*50)
print("DỰ ĐOÁN CHO BỆNH NHÂN MỚI")
print("="*50)
print(f"Kết quả: {'CÓ TIỂU ĐƯỜNG' if prediction[0] == 1 else 'KHÔNG CÓ TIỂU ĐƯỜNG'}")
print(f"Xác suất KHÔNG có tiểu đường: {probability[0][0]:.2%}")
print(f"Xác suất CÓ tiểu đường: {probability[0][1]:.2%}")
print("="*50)
```

---

## Dự Đoán Sinh Viên Có Tốt Nghiệp Đúng Hạn

### Mục Tiêu
Dự đoán xem sinh viên có tốt nghiệp đúng hạn (4 năm) hay không dựa trên các yếu tố học tập.

### Tạo Dữ Liệu Mẫu

```python
import pandas as pd
import numpy as np
from random_forest import RandomForestClassificationModel
from sklearn.model_selection import train_test_split

# Tạo dữ liệu sinh viên mẫu
np.random.seed(42)
n_students = 1000

student_data = pd.DataFrame({
    'gpa_year1': np.random.uniform(2.0, 4.0, n_students),
    'gpa_year2': np.random.uniform(2.0, 4.0, n_students),
    'attendance_rate': np.random.uniform(0.5, 1.0, n_students),
    'credits_per_semester': np.random.randint(12, 20, n_students),
    'part_time_job_hours': np.random.randint(0, 30, n_students),
    'study_hours_per_week': np.random.randint(5, 40, n_students),
    'clubs_activities': np.random.randint(0, 5, n_students),
    'scholarship': np.random.choice([0, 1], n_students, p=[0.7, 0.3]),
    'commute_time_minutes': np.random.randint(10, 120, n_students),
})

# Tạo target dựa trên logic
# Tốt nghiệp đúng hạn phụ thuộc vào GPA, attendance, credits
student_data['graduate_on_time'] = (
    (student_data['gpa_year1'] > 2.5) &
    (student_data['gpa_year2'] > 2.5) &
    (student_data['attendance_rate'] > 0.75) &
    (student_data['credits_per_semester'] >= 15)
).astype(int)

# Thêm một số noise
noise_indices = np.random.choice(n_students, size=int(n_students * 0.1), replace=False)
student_data.loc[noise_indices, 'graduate_on_time'] = 1 - student_data.loc[noise_indices, 'graduate_on_time']

print("Phân bố nhãn:")
print(student_data['graduate_on_time'].value_counts())
print(f"\nTỉ lệ: {student_data['graduate_on_time'].value_counts(normalize=True)}")
```

### Huấn Luyện Mô Hình

```python
# Tách features và target
X = student_data.drop('graduate_on_time', axis=1)
y = student_data['graduate_on_time']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Tạo và huấn luyện mô hình
model = RandomForestClassificationModel(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train, feature_names=X.columns.tolist())

# Đánh giá
print("\nĐánh giá mô hình:")
metrics = model.evaluate(X_test, y_test)
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# Feature importance
print("\nYếu tố ảnh hưởng đến khả năng tốt nghiệp đúng hạn:")
importance_df = model.get_feature_importance()
print(importance_df)

# Vẽ biểu đồ
model.plot_feature_importance(top_n=9)
```

### Dự Đoán và Phân Tích

```python
# Dự đoán cho sinh viên mới
new_students = pd.DataFrame({
    'gpa_year1': [3.5, 2.8, 2.2],
    'gpa_year2': [3.6, 2.9, 2.5],
    'attendance_rate': [0.95, 0.85, 0.70],
    'credits_per_semester': [18, 15, 12],
    'part_time_job_hours': [10, 20, 25],
    'study_hours_per_week': [25, 15, 10],
    'clubs_activities': [2, 1, 0],
    'scholarship': [1, 0, 0],
    'commute_time_minutes': [30, 60, 90]
})

predictions = model.predict(new_students)
probabilities = model.predict_proba(new_students)

print("\n" + "="*70)
print("DỰ ĐOÁN CHO SINH VIÊN MỚI")
print("="*70)
for i in range(len(new_students)):
    print(f"\nSinh viên {i+1}:")
    print(f"  GPA: {new_students.iloc[i]['gpa_year2']:.2f}")
    print(f"  Attendance: {new_students.iloc[i]['attendance_rate']:.0%}")
    print(f"  Dự đoán: {'TỐT NGHIỆP ĐÚNG HẠN' if predictions[i] == 1 else 'KHÔNG TỐT NGHIỆP ĐÚNG HẠN'}")
    print(f"  Xác suất tốt nghiệp đúng hạn: {probabilities[i][1]:.2%}")
print("="*70)
```

---

## Phân Tích A/B Testing

### Mục Tiêu
Dự đoán xem người dùng có chuyển đổi (conversion) sau khi xem phiên bản A hoặc B của trang web.

### Code Hoàn Chỉnh

```python
import pandas as pd
import numpy as np
from random_forest import RandomForestClassificationModel
from sklearn.model_split import train_test_split

# Tạo dữ liệu A/B test
np.random.seed(42)
n_users = 5000

ab_data = pd.DataFrame({
    'variant': np.random.choice(['A', 'B'], n_users),
    'time_on_page_seconds': np.random.exponential(scale=120, size=n_users),
    'pages_visited': np.random.poisson(lam=3, size=n_users),
    'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_users, p=[0.6, 0.3, 0.1]),
    'hour_of_day': np.random.randint(0, 24, n_users),
    'is_returning_user': np.random.choice([0, 1], n_users, p=[0.7, 0.3]),
    'referral_source': np.random.choice(['organic', 'paid', 'social', 'direct'], n_users, p=[0.3, 0.3, 0.2, 0.2])
})

# Tạo conversion (B tốt hơn A một chút)
conversion_prob = 0.10 # base rate
ab_data['conversion_prob'] = conversion_prob
ab_data.loc[ab_data['variant'] == 'B', 'conversion_prob'] += 0.03  # B có conversion cao hơn 3%
ab_data.loc[ab_data['is_returning_user'] == 1, 'conversion_prob'] += 0.05
ab_data.loc[ab_data['time_on_page_seconds'] > 180, 'conversion_prob'] += 0.08

ab_data['converted'] = (np.random.random(n_users) < ab_data['conversion_prob']).astype(int)
ab_data = ab_data.drop('conversion_prob', axis=1)

print("Conversion rate by variant:")
print(ab_data.groupby('variant')['converted'].mean())

# One-hot encoding
ab_encoded = pd.get_dummies(ab_data, columns=['variant', 'device', 'referral_source'], drop_first=True)

# Tách features và target
X = ab_encoded.drop('converted', axis=1)
y = ab_encoded['converted']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Huấn luyện mô hình
model = RandomForestClassificationModel(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.model.set_params(class_weight='balanced')
model.fit(X_train, y_train, feature_names=X.columns.tolist())

# Đánh giá
print("\nĐánh giá mô hình:")
metrics = model.evaluate(X_test, y_test)
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# Feature importance
print("\nYếu tố ảnh hưởng đến conversion:")
importance_df = model.get_feature_importance(top_n=10)
print(importance_df)
model.plot_feature_importance(top_n=10)
```

---

## Tổng Kết

Các ví dụ trên minh họa:
- ✅ Cách xử lý các loại dữ liệu khác nhau
- ✅ Xử lý imbalanced data
- ✅ Feature engineering và preprocessing
- ✅ Hyperparameter tuning
- ✅ Feature importance analysis
- ✅ Dự đoán và diễn giải kết quả

Để biết thêm chi tiết, xem:
- [Random Forest Overview](random_forest_overview.md)
- [Usage Guide](random_forest_usage_guide.md)
