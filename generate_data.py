# =============================================================================
# generate_data.py - Tạo dataset khách hàng giả lập
# =============================================================================
# Mô tả: Tạo dataset đơn giản với 800 dòng, 5 cột chính
#         phục vụ bài toán phân cụm khách hàng bằng KMeans
# Tác giả: Nhóm 08 - Môn Khai Phá Dữ Liệu
# =============================================================================

import pandas as pd
import numpy as np
import random
import os

# Đặt seed để kết quả có thể tái tạo (reproducible)
np.random.seed(42)
random.seed(42)


def generate_customer_data(n_customers=800):
    """
    Tạo dataset khách hàng giả lập đơn giản.

    Dataset gồm 5 cột:
    - Age: Tuổi khách hàng (18-65)
    - Gender: Giới tính (Nam/Nữ)
    - Income: Thu nhập hàng tháng (triệu VNĐ)
    - SpendingScore: Điểm chi tiêu (1-100)
    - PurchaseFrequency: Tần suất mua hàng (lần/tháng)

    Parameters:
    -----------
    n_customers : int
        Số lượng khách hàng cần tạo (mặc định: 800)

    Returns:
    --------
    pd.DataFrame
        DataFrame chứa thông tin khách hàng
    """
    print(f"🔄 Đang tạo dataset với {n_customers} khách hàng...")

    # --- Tạo dữ liệu ---
    data = {
        # ID khách hàng: KH001 -> KH800
        'CustomerID': [f'KH{str(i).zfill(4)}' for i in range(1, n_customers + 1)],

        # Giới tính: phân phối ngẫu nhiên
        'Gender': [random.choice(['Nam', 'Nữ']) for _ in range(n_customers)],

        # Tuổi: phân phối chuẩn, trung bình 35, giới hạn [18, 65]
        'Age': np.clip(
            np.random.normal(loc=35, scale=12, size=n_customers).astype(int),
            18, 65
        ),

        # Thu nhập hàng tháng (triệu VNĐ): phân phối log-normal
        'Income': np.round(
            np.random.lognormal(mean=2.5, sigma=0.6, size=n_customers), 1
        ),

        # Điểm chi tiêu (1-100): phân phối chuẩn
        'SpendingScore': np.clip(
            np.random.normal(loc=50, scale=20, size=n_customers).astype(int),
            1, 100
        ),

        # Tần suất mua hàng (lần/tháng): phân phối Poisson
        'PurchaseFrequency': np.clip(
            np.random.poisson(lam=5, size=n_customers),
            0, 25
        ),
    }

    # Tạo DataFrame
    df = pd.DataFrame(data)

    # --- Thêm missing values (~2%) để mô phỏng dữ liệu thực tế ---
    n_missing = int(n_customers * 0.02)
    for col in ['Age', 'Income', 'SpendingScore']:
        missing_idx = random.sample(range(n_customers), n_missing)
        df.loc[missing_idx, col] = np.nan

    # --- Thêm dòng trùng lặp (~1%) ---
    n_dup = int(n_customers * 0.01)
    dup_rows = df.sample(n=n_dup, random_state=42)
    df = pd.concat([df, dup_rows], ignore_index=True)

    print(f"✅ Dataset đã tạo thành công!")
    print(f"   - Số dòng: {len(df)}")
    print(f"   - Số cột: {len(df.columns)}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Duplicates: {df.duplicated().sum()}")

    return df


def save_dataset(df, filename='customers.csv'):
    """Lưu dataset ra file CSV trong thư mục data/"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"💾 Dataset đã lưu tại: {filepath}")
    return filepath


# =============================================================================
# MAIN - Chạy file này để tạo dataset
# =============================================================================
if __name__ == '__main__':
    # Tạo dataset 800 khách hàng
    df = generate_customer_data(n_customers=800)

    # Lưu file CSV
    save_dataset(df)

    # Hiển thị thông tin
    print("\n📊 Thông tin dataset:")
    print("-" * 40)
    print(df.info())
    print("\n📈 Thống kê mô tả:")
    print("-" * 40)
    print(df.describe())
    print("\n🔍 5 dòng đầu tiên:")
    print("-" * 40)
    print(df.head())
