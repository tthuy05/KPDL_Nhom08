# =============================================================================
# generate_data.py - Tạo dataset khách hàng giả lập
# =============================================================================
# Mô tả: Sử dụng thư viện Faker và NumPy để tạo dataset khách hàng
#         với 2000 dòng dữ liệu mô phỏng thực tế.
# Tác giả: Nhóm 08 - Môn Khai Phá Dữ Liệu
# =============================================================================

import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Khởi tạo Faker với locale tiếng Việt
fake = Faker('vi_VN')

# Đặt seed để kết quả có thể tái tạo (reproducible)
np.random.seed(42)
random.seed(42)
Faker.seed(42)


def generate_customer_data(n_customers=2000):
    """
    Tạo dataset khách hàng giả lập.

    Parameters:
    -----------
    n_customers : int
        Số lượng khách hàng cần tạo (mặc định: 2000)

    Returns:
    --------
    pd.DataFrame
        DataFrame chứa thông tin khách hàng
    """

    print(f"🔄 Đang tạo dataset với {n_customers} khách hàng...")

    # --- Định nghĩa các giá trị có thể ---
    genders = ['Nam', 'Nữ']
    membership_levels = ['Bronze', 'Silver', 'Gold', 'Platinum']
    preferred_categories = [
        'Thời trang', 'Điện tử', 'Thực phẩm',
        'Mỹ phẩm', 'Gia dụng', 'Sách', 'Thể thao'
    ]

    # --- Tạo từng cột dữ liệu ---
    data = {
        # ID khách hàng: KH0001 -> KH2000
        'CustomerID': [f'KH{str(i).zfill(4)}' for i in range(1, n_customers + 1)],

        # Giới tính: phân phối ngẫu nhiên
        'Gender': [random.choice(genders) for _ in range(n_customers)],

        # Tuổi: phân phối chuẩn, trung bình 35, std 12, giới hạn [18, 70]
        'Age': np.clip(
            np.random.normal(loc=35, scale=12, size=n_customers).astype(int),
            18, 70
        ),

        # Thu nhập hàng tháng (triệu VNĐ): phân phối log-normal
        'Income': np.round(
            np.random.lognormal(mean=2.5, sigma=0.6, size=n_customers), 1
        ),

        # Điểm chi tiêu (0-100): phân phối chuẩn
        'SpendingScore': np.clip(
            np.random.normal(loc=50, scale=20, size=n_customers).astype(int),
            1, 100
        ),

        # Tần suất mua hàng (số lần/tháng): phân phối Poisson
        'PurchaseFrequency': np.clip(
            np.random.poisson(lam=5, size=n_customers),
            0, 30
        ),

        # Giá trị đơn hàng trung bình (triệu VNĐ)
        'AverageOrderValue': np.round(
            np.random.exponential(scale=2.0, size=n_customers), 2
        ),

        # Số ngày kể từ lần mua cuối: phân phối exponential
        'LastPurchaseDays': np.clip(
            np.random.exponential(scale=30, size=n_customers).astype(int),
            0, 365
        ),

        # Cấp độ thành viên
        'MembershipLevel': random.choices(
            membership_levels,
            weights=[40, 30, 20, 10],  # Bronze phổ biến nhất
            k=n_customers
        ),

        # Danh mục yêu thích
        'PreferredCategory': [random.choice(preferred_categories) for _ in range(n_customers)],
    }

    # --- Tạo DataFrame ---
    df = pd.DataFrame(data)

    # --- Thêm một số missing values để mô phỏng thực tế ---
    # Khoảng 2% dữ liệu bị thiếu
    n_missing = int(n_customers * 0.02)
    missing_cols = ['Age', 'Income', 'SpendingScore', 'AverageOrderValue']

    for col in missing_cols:
        missing_indices = random.sample(range(n_customers), n_missing)
        df.loc[missing_indices, col] = np.nan

    # --- Thêm một số dòng trùng lặp để mô phỏng thực tế ---
    n_duplicates = int(n_customers * 0.01)  # 1% trùng lặp
    duplicate_rows = df.sample(n=n_duplicates, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    print(f"✅ Dataset đã tạo thành công!")
    print(f"   - Số dòng: {len(df)}")
    print(f"   - Số cột: {len(df.columns)}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Duplicates: {df.duplicated().sum()}")

    return df


def save_dataset(df, filename='customers.csv'):
    """
    Lưu dataset ra file CSV.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần lưu
    filename : str
        Tên file CSV đầu ra
    """
    # Tạo thư mục data nếu chưa có
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"💾 Dataset đã lưu tại: {filepath}")

    return filepath


# =============================================================================
# MAIN - Chạy trực tiếp file này để tạo dataset
# =============================================================================
if __name__ == '__main__':
    # Tạo dataset
    df = generate_customer_data(n_customers=2000)

    # Lưu ra file CSV
    save_dataset(df)

    # Hiển thị thông tin dataset
    print("\n📊 Thông tin dataset:")
    print("-" * 50)
    print(df.info())
    print("\n📈 Thống kê mô tả:")
    print("-" * 50)
    print(df.describe())
    print("\n🔍 5 dòng đầu tiên:")
    print("-" * 50)
    print(df.head())
