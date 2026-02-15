# Random Forest Classification - Tổng Quan

## Giới Thiệu

Random Forest (Rừng Ngẫu Nhiên) là một thuật toán học máy mạnh mẽ, thuộc nhóm **ensemble learning** (học tập tổ hợp). Thuật toán này hoạt động bằng cách xây dựng nhiều cây quyết định (decision trees) trong quá trình huấn luyện và đưa ra kết quả phân loại dựa trên sự kết hợp của các cây này.

## Nguyên Lý Hoạt Động

### 1. Bootstrap Aggregating (Bagging)
- Random Forest sử dụng kỹ thuật **bootstrap sampling** để tạo ra nhiều tập dữ liệu con từ tập dữ liệu gốc
- Mỗi tập con được lấy mẫu ngẫu nhiên có hoàn lại (random sampling with replacement)
- Mỗi cây quyết định được huấn luyện trên một tập dữ liệu con khác nhau

### 2. Xây Dựng Cây Quyết Định
- Với mỗi tập dữ liệu bootstrap, một cây quyết định được xây dựng
- Tại mỗi nút của cây, thuật toán chỉ xem xét một tập con ngẫu nhiên các đặc trưng (features) để tìm phân chia tốt nhất
- Các cây được phép phát triển đến độ sâu tối đa mà không cần pruning (cắt tỉa)

### 3. Kết Hợp Dự Đoán (Voting)
Khi dự đoán cho một mẫu mới:
- **Classification**: Mỗi cây đưa ra một "phiếu bầu" cho một lớp, lớp nhận được nhiều phiếu nhất được chọn
- Công thức: `Predicted Class = mode{h₁(x), h₂(x), ..., hₙ(x)}`

Trong đó:
- `hᵢ(x)` là dự đoán của cây thứ i
- `mode{}` là giá trị xuất hiện nhiều nhất (majority voting)

## Tham Số Quan Trọng

### 1. `n_estimators` (Số lượng cây)
- **Mặc định**: 100
- **Ý nghĩa**: Số lượng cây quyết định trong rừng
- **Ảnh hưởng**: 
  - Càng nhiều cây → độ chính xác càng cao nhưng thời gian huấn luyện lâu hơn
  - Thường từ 100-500 cây là đủ cho hầu hết các bài toán

### 2. `max_depth` (Độ sâu tối đa)
- **Mặc định**: None (không giới hạn)
- **Ý nghĩa**: Độ sâu tối đa của mỗi cây quyết định
- **Ảnh hưởng**:
  - Cây sâu → học chi tiết hơn nhưng dễ overfitting
  - Cây nông → tổng quát hóa tốt nhưng có thể underfitting

### 3. `min_samples_split` (Số mẫu tối thiểu để phân chia)
- **Mặc định**: 2
- **Ý nghĩa**: Số mẫu tối thiểu cần có tại một nút để được phép phân chia tiếp
- **Ảnh hưởng**: Giá trị cao → cây đơn giản hơn, giảm overfitting

### 4. `min_samples_leaf` (Số mẫu tối thiểu tại lá)
- **Mặc định**: 1
- **Ý nghĩa**: Số mẫu tối thiểu phải có tại một nút lá
- **Ảnh hưởng**: Giá trị cao → tạo ra các lá "mượt mại" hơn

### 5. `max_features` (Số đặc trưng tối đa)
- **Mặc định**: 'sqrt' (căn bậc hai của tổng số đặc trưng)
- **Tùy chọn**:
  - `'sqrt'`: √(n_features)
  - `'log2'`: log₂(n_features)
  - Số nguyên: số lượng đặc trưng cụ thể
  - `None`: sử dụng tất cả đặc trưng

## Ưu Điểm

### 1. Hiệu Suất Cao
- Độ chính xác cao trên nhiều loại dữ liệu khác nhau
- Hoạt động tốt cả trên dữ liệu tuyến tính và phi tuyến

### 2. Chống Overfitting
- Việc kết hợp nhiều cây giúp giảm thiểu overfitting so với một cây quyết định đơn lẻ
- Tính ngẫu nhiên trong việc chọn mẫu và đặc trưng làm tăng tính tổng quát

### 3. Xử Lý Dữ Liệu Tốt
- Xử lý được cả dữ liệu số (numerical) và phân loại (categorical)
- Không cần chuẩn hóa dữ liệu (feature scaling)
- Hoạt động tốt với dữ liệu thiếu (missing values)

### 4. Đánh Giá Tầm Quan Trọng Đặc Trưng
- Cung cấp thông tin về độ quan trọng của từng đặc trưng
- Hữu ích cho việc lựa chọn đặc trưng (feature selection)

### 5. Xử Lý Dữ Liệu Lớn
- Có khả năng xử lý tập dữ liệu với số chiều cao (high-dimensional data)
- Hỗ trợ song song hóa (parallel processing) → tăng tốc độ huấn luyện

## Nhược Điểm

### 1. Tốn Bộ Nhớ
- Cần lưu trữ nhiều cây quyết định
- Với tập dữ liệu lớn, có thể tiêu tốn nhiều RAM

### 2. Thời Gian Huấn Luyện
- Huấn luyện nhiều cây mất thời gian
- Tuy nhiên, có thể song song hóa để tăng tốc

### 3. Khó Diễn Giải
- Không trực quan như một cây quyết định đơn lẻ
- Khó hiểu được "lý do" đằng sau mỗi dự đoán

### 4. Không Tối Ưu Cho Dữ Liệu Thưa
- Hiệu suất kém trên dữ liệu có nhiều đặc trưng thưa (sparse features)
- Ví dụ: dữ liệu văn bản với TF-IDF vectors

## Ứng Dụng Thực Tế

### 1. Y Tế
- Chẩn đoán bệnh dựa trên triệu chứng
- Dự đoán nguy cơ mắc bệnh
- Phân loại hình ảnh y tế

### 2. Tài Chính
- Phát hiện gian lận thẻ tín dụng
- Đánh giá rủi ro tín dụng
- Dự đoán vỡ nợ

### 3. Thương Mại Điện Tử
- Hệ thống gợi ý sản phẩm
- Phân loại khách hàng
- Dự đoán churn (khách hàng rời bỏ)

### 4. Nhận Dạng Hình Ảnh
- Phân loại đối tượng
- Nhận dạng khuôn mặt
- Phát hiện đối tượng

## So Sánh Với Các Thuật Toán Khác

| Tiêu Chí | Random Forest | Decision Tree | SVM | Neural Networks |
|----------|--------------|---------------|-----|-----------------|
| Độ chính xác | Cao | Trung bình | Cao | Rất cao |
| Tốc độ huấn luyện | Trung bình | Nhanh | Chậm | Rất chậm |
| Khả năng diễn giải | Thấp | Cao | Thấp | Rất thấp |
| Chống overfitting | Tốt | Kém | Tốt | Cần điều chỉnh |
| Xử lý dữ liệu lớn | Tốt | Tốt | Kém | Tốt |

## Khi Nào Nên Sử Dụng Random Forest?

### ✅ Nên Dùng Khi:
- Cần mô hình chính xác cao mà không cần quá phức tạp
- Dữ liệu có nhiều đặc trưng
- Muốn biết đặc trưng nào quan trọng
- Có dữ liệu thiếu hoặc nhiễu
- Không cần giải thích chi tiết từng dự đoán

### ❌ Không Nên Dùng Khi:
- Cần mô hình dễ giải thích (dùng Decision Tree thay thế)
- Dữ liệu có cấu trúc tuần tự (dùng RNN/LSTM)
- Bộ nhớ hạn chế nghiêm ngặt
- Cần dự đoán thời gian thực với độ trễ cực thấp

## Tài Liệu Tham Khảo

1. Breiman, L. (2001). "Random Forests". Machine Learning. 45 (1): 5–32.
2. Scikit-learn Documentation: [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. Hastie, T., Tibshirani, R., Friedman, J. (2009). "The Elements of Statistical Learning"

---

**Ghi chú**: Tài liệu này cung cấp cái nhìn tổng quan về Random Forest Classification. Để biết cách sử dụng chi tiết, vui lòng tham khảo file `random_forest_usage_guide.md`.
