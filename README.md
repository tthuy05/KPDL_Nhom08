# 📊 Phân Cụm Khách Hàng - Đồ Án Khai Phá Dữ Liệu

## Đề tài
**Ứng dụng kỹ thuật gom nhóm trong khai thác dữ liệu khách hàng để phân khúc thị trường**

**Nhóm 08** — Môn Khai Phá Dữ Liệu

---

## 📁 Cấu trúc Project

```
KPDL_Nhom08/
├── data/
│   └── customers.csv          # Dataset khách hàng (2000+ dòng)
├── generate_data.py           # Tạo dataset giả lập
├── preprocessing.py           # Tiền xử lý dữ liệu
├── clustering.py              # Thuật toán gom cụm
├── visualization.py           # Trực quan hóa dữ liệu
├── app.py                     # Dashboard Streamlit
├── requirements.txt           # Thư viện cần cài
└── README.md                  # Hướng dẫn (file này)
```

---

## 🚀 Hướng dẫn cài đặt và chạy

### Bước 1: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Tạo dataset

```bash
python generate_data.py
```

→ File `data/customers.csv` sẽ được tạo với ~2000 dòng dữ liệu.

### Bước 3: Chạy Dashboard

```bash
streamlit run app.py
```

→ Mở trình duyệt tại `http://localhost:8501`

### (Tùy chọn) Chạy từng module riêng

```bash
python preprocessing.py     # Test tiền xử lý
python clustering.py        # Test thuật toán clustering
python visualization.py     # Tạo biểu đồ
```

---

## 🔧 Công nghệ sử dụng

| Thư viện | Mục đích |
|----------|----------|
| **Pandas** | Xử lý dữ liệu dạng bảng |
| **NumPy** | Tính toán số học |
| **Matplotlib** | Vẽ biểu đồ cơ bản |
| **Seaborn** | Vẽ biểu đồ nâng cao |
| **Scikit-learn** | Thuật toán ML & đánh giá |
| **Streamlit** | Dashboard web tương tác |
| **Faker** | Sinh dữ liệu giả lập |

---

## 📋 Mô tả Dataset

| Cột | Mô tả | Kiểu |
|-----|--------|------|
| CustomerID | Mã khách hàng | String |
| Gender | Giới tính (Nam/Nữ) | Categorical |
| Age | Tuổi (18-70) | Numeric |
| Income | Thu nhập hàng tháng (triệu VNĐ) | Numeric |
| SpendingScore | Điểm chi tiêu (1-100) | Numeric |
| PurchaseFrequency | Tần suất mua (lần/tháng) | Numeric |
| AverageOrderValue | Giá trị đơn hàng TB (triệu VNĐ) | Numeric |
| LastPurchaseDays | Số ngày từ lần mua cuối | Numeric |
| MembershipLevel | Cấp thành viên | Categorical |
| PreferredCategory | Danh mục yêu thích | Categorical |

---

## 🤖 Thuật toán Clustering

### 1. KMeans
- Chia dữ liệu thành K cụm dựa trên khoảng cách Euclidean
- Sử dụng Elbow Method để tìm K tối ưu

### 2. DBSCAN
- Gom cụm dựa trên mật độ điểm dữ liệu
- Tự động phát hiện số cụm và noise

### 3. Agglomerative Clustering
- Phân cụm phân cấp (bottom-up)
- Hỗ trợ nhiều phương pháp linkage

---

## 📊 Đánh giá mô hình

| Phương pháp | Mô tả |
|-------------|--------|
| **Elbow Method** | Tìm K tối ưu qua điểm "khuỷu tay" của Inertia |
| **Silhouette Score** | Đo chất lượng cụm (-1 đến 1, càng cao càng tốt) |
| **Calinski-Harabasz** | Tỷ lệ phương sai giữa/trong cụm (càng cao càng tốt) |

---

## 🏷️ Phân loại nhóm khách hàng

| Nhóm | Đặc điểm | Chiến lược Marketing |
|------|----------|---------------------|
| 💎 **Khách VIP** | Thu nhập cao, chi tiêu cao | Chương trình loyalty, ưu đãi độc quyền |
| ⭐ **Khách tiềm năng** | Thu nhập & chi tiêu trung bình | Upselling, khuyến mãi có mục tiêu |
| 💰 **Khách tiết kiệm** | Thu nhập & chi tiêu thấp | Sản phẩm giá tốt, combo tiết kiệm |
| 🏷️ **Khách săn giảm giá** | Mua nhiều, giá trị đơn thấp | Flash sale, voucher giảm giá |
| 😴 **Khách không hoạt động** | Lâu không mua hàng | Email re-engagement, ưu đãi quay lại |

---

## 📝 Nhận xét kết quả

1. **KMeans** cho kết quả phân cụm rõ ràng nhất khi K=4, phù hợp cho bài toán phân khúc khách hàng.
2. **DBSCAN** phát hiện được các điểm noise (outliers), hữu ích để nhận diện khách hàng bất thường.
3. **Agglomerative** cho kết quả tương tự KMeans, phù hợp khi cần phân tích phân cấp.
4. Silhouette Score trung bình ~0.2-0.4 cho thấy các cụm có sự phân tách hợp lý.
5. Thu nhập và Điểm chi tiêu là 2 đặc trưng quan trọng nhất để phân cụm.

---

## 👥 Thành viên nhóm

| STT | Họ tên | MSSV |
|-----|--------|------|
| 1 | [Tên SV 1] | [MSSV] |
| 2 | [Tên SV 2] | [MSSV] |
| 3 | [Tên SV 3] | [MSSV] |
