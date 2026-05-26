# 📊 Phân Cụm Khách Hàng bằng KMeans

## Đề tài
**Ứng dụng kỹ thuật gom nhóm (KMeans) trong phân cụm khách hàng để phân khúc thị trường**

**Nhóm 08** — Môn Khai Phá Dữ Liệu

---

## 📁 Cấu trúc Project

```
KPDL_Nhom08/
├── data/
│   ├── customer_data_with_churn.csv # Dataset Kaggle gốc (~2000 dòng)
│   └── customers_kaggle.csv         # Dataset đã chuẩn hóa, dùng mặc định
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

### Bước 2: Chạy Dashboard

```bash
streamlit run app.py
```

→ Mở trình duyệt tại `http://localhost:8501`

---

## 📂 Nguồn dữ liệu

Dataset được lấy từ **Kaggle**. File `customer_data_with_churn.csv` là dữ liệu gốc; file
`customers_kaggle.csv` là dữ liệu đã chọn thuộc tính và đổi tên cột để dùng mặc định.

Dashboard sử dụng file đã chuẩn hóa hoặc CSV có các cột:
`CustomerID`, `Gender`, `Age`, `Income`, `SpendingScore`, `PurchaseFrequency`.

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
| Age | Tuổi (18-69) | Numeric |
| Income | Thu nhập hàng tháng | Numeric |
| SpendingScore | Điểm chi tiêu (1-100) | Numeric |
| PurchaseFrequency | Tần suất mua (lần/tháng) | Numeric |

---

## 🤖 Thuật toán KMeans

**Nguyên lý hoạt động:**
1. Chọn ngẫu nhiên K tâm cụm (centroids)
2. Gán mỗi điểm vào cụm có tâm gần nhất (khoảng cách Euclidean)
3. Cập nhật tâm cụm = trung bình các điểm trong cụm
4. Lặp lại bước 2-3 cho đến khi hội tụ

**Tìm K phù hợp:** Khảo sát `K=2..10` bằng Elbow Method và Silhouette Score.
Với dataset mặc định, `K=8` đạt Silhouette Score cao nhất trong phạm vi khảo sát.

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
| 1 | Trần Lê Anh Tuấn | 2001230858 |
| 2 | Nguyễn Đẩu Thủy | 2001230951 |
