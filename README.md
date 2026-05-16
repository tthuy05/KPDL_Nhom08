# 📊 Phân Cụm Khách Hàng bằng KMeans

## Đề tài
**Ứng dụng kỹ thuật gom nhóm (KMeans) trong phân cụm khách hàng để phân khúc thị trường**

**Nhóm 08** — Môn Khai Phá Dữ Liệu

---

## 📁 Cấu trúc Project

```
KPDL_Nhom08/
├── data/
│   └── customers.csv          # Dataset khách hàng (~800 dòng)
├── generate_data.py           # Tạo dataset giả lập
├── preprocessing.py           # Tiền xử lý dữ liệu
├── clustering.py              # Thuật toán KMeans
├── visualization.py           # Trực quan hóa (Histogram, Heatmap, Scatter)
├── app.py                     # Dashboard Streamlit
├── requirements.txt           # Thư viện cần cài
└── README.md                  # Hướng dẫn (file này)
```

---

## 🚀 Hướng dẫn chạy

### Bước 1: Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Tạo dataset

```bash
python generate_data.py
```

### Bước 3: Chạy Dashboard

```bash
streamlit run app.py
```

→ Mở trình duyệt tại `http://localhost:8501`

---

## 🔧 Công nghệ sử dụng

| Thư viện | Mục đích |
|----------|----------|
| **Pandas** | Xử lý dữ liệu dạng bảng |
| **NumPy** | Tính toán số học |
| **Matplotlib** | Vẽ biểu đồ |
| **Seaborn** | Vẽ heatmap |
| **Scikit-learn** | Thuật toán KMeans & đánh giá |
| **Streamlit** | Dashboard web tương tác |

---

## 📋 Mô tả Dataset

| Cột | Mô tả | Kiểu |
|-----|--------|------|
| CustomerID | Mã khách hàng | String |
| Gender | Giới tính (Nam/Nữ) | Categorical |
| Age | Tuổi (18-65) | Numeric |
| Income | Thu nhập hàng tháng (triệu VNĐ) | Numeric |
| SpendingScore | Điểm chi tiêu (1-100) | Numeric |
| PurchaseFrequency | Tần suất mua (lần/tháng) | Numeric |

---

## 🤖 Thuật toán KMeans

**Nguyên lý hoạt động:**
1. Chọn ngẫu nhiên K tâm cụm (centroids)
2. Gán mỗi điểm vào cụm có tâm gần nhất (khoảng cách Euclidean)
3. Cập nhật tâm cụm = trung bình các điểm trong cụm
4. Lặp lại bước 2-3 cho đến khi hội tụ

**Tìm K tối ưu:** Sử dụng Elbow Method + Silhouette Score

---

## 📊 Visualization

| Biểu đồ | Mục đích |
|----------|----------|
| **Histogram** | Xem phân phối dữ liệu từng cột |
| **Heatmap** | Xem tương quan giữa các biến |
| **Scatter Plot** | Xem kết quả phân cụm trực quan |

---

## 🏷️ Phân loại nhóm khách hàng

| Nhóm | Đặc điểm |
|------|----------|
| 💎 **Khách VIP** | Thu nhập cao, chi tiêu cao |
| ⭐ **Khách tiềm năng** | Thu nhập & chi tiêu trung bình-khá |
| 🛒 **Khách ít mua hàng** | Tần suất mua hàng thấp |
| 💰 **Khách chi tiêu thấp** | Điểm chi tiêu thấp |

---

## 👥 Thành viên nhóm

| STT | Họ tên | MSSV |
|-----|--------|------|
| 1 | [Tên SV 1] | [MSSV] |
| 2 | [Tên SV 2] | [MSSV] |
