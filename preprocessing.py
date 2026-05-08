# =============================================================================
# preprocessing.py - Tiền xử lý dữ liệu khách hàng
# =============================================================================
# Mô tả: Thực hiện các bước tiền xử lý dữ liệu bao gồm:
#         - Xử lý missing values
#         - Xóa dữ liệu trùng lặp
#         - Label Encoding cho biến phân loại
#         - Chuẩn hóa dữ liệu (StandardScaler)
# Tác giả: Nhóm 08 - Môn Khai Phá Dữ Liệu
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Đọc dữ liệu từ file CSV.

    Parameters:
    -----------
    filepath : str
        Đường dẫn đến file CSV

    Returns:
    --------
    pd.DataFrame
        DataFrame chứa dữ liệu khách hàng
    """
    print(f"📂 Đang đọc dữ liệu từ: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   - Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")
    return df


def handle_missing_values(df):
    """
    Xử lý giá trị bị thiếu (missing values).

    Chiến lược:
    - Cột số (numeric): điền bằng giá trị trung vị (median)
      → Median ít bị ảnh hưởng bởi outliers hơn mean
    - Cột phân loại (categorical): điền bằng giá trị phổ biến nhất (mode)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý

    Returns:
    --------
    pd.DataFrame
        DataFrame đã xử lý missing values
    """
    print("\n🔧 Xử lý Missing Values:")
    print("-" * 40)

    # Đếm missing values trước khi xử lý
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]

    if len(missing_cols) == 0:
        print("   ✅ Không có missing values!")
        return df

    for col in missing_cols.index:
        count = missing_cols[col]
        if df[col].dtype in ['float64', 'int64']:
            # Cột số: điền bằng median
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   - {col}: {count} missing → điền median = {median_val}")
        else:
            # Cột phân loại: điền bằng mode
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"   - {col}: {count} missing → điền mode = '{mode_val}'")

    print(f"   ✅ Đã xử lý xong {missing_before.sum()} missing values!")
    return df


def remove_duplicates(df):
    """
    Xóa các dòng dữ liệu trùng lặp.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý

    Returns:
    --------
    pd.DataFrame
        DataFrame đã xóa trùng lặp
    """
    print("\n🔧 Xử lý Duplicates:")
    print("-" * 40)

    n_duplicates = df.duplicated().sum()

    if n_duplicates == 0:
        print("   ✅ Không có dữ liệu trùng lặp!")
        return df

    df = df.drop_duplicates().reset_index(drop=True)
    print(f"   - Đã xóa {n_duplicates} dòng trùng lặp")
    print(f"   - Kích thước mới: {df.shape[0]} dòng")

    return df


def encode_categorical(df):
    """
    Mã hóa biến phân loại (Label Encoding).

    Label Encoding chuyển đổi giá trị text thành số:
    Ví dụ: 'Bronze' → 0, 'Gold' → 1, 'Platinum' → 2, 'Silver' → 3

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần mã hóa

    Returns:
    --------
    tuple: (pd.DataFrame, dict)
        - DataFrame đã mã hóa
        - Dictionary chứa các LabelEncoder để decode sau này
    """
    print("\n🔧 Label Encoding:")
    print("-" * 40)

    # Các cột phân loại cần mã hóa
    categorical_cols = ['Gender', 'MembershipLevel', 'PreferredCategory']
    encoders = {}

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_Encoded'] = le.fit_transform(df[col])
            encoders[col] = le

            # Hiển thị mapping
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"   - {col}: {mapping}")

    print(f"   ✅ Đã mã hóa {len(encoders)} cột phân loại!")
    return df, encoders


def scale_features(df, feature_cols=None):
    """
    Chuẩn hóa dữ liệu bằng StandardScaler.

    StandardScaler: chuyển đổi dữ liệu về phân phối chuẩn
    với mean = 0 và std = 1.
    Công thức: z = (x - mean) / std

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần chuẩn hóa
    feature_cols : list, optional
        Danh sách cột cần chuẩn hóa

    Returns:
    --------
    tuple: (np.ndarray, StandardScaler, list)
        - Mảng dữ liệu đã chuẩn hóa
        - Đối tượng StandardScaler (để inverse_transform sau này)
        - Danh sách tên cột đã chuẩn hóa
    """
    print("\n🔧 Chuẩn hóa dữ liệu (StandardScaler):")
    print("-" * 40)

    # Nếu không chỉ định cột, chọn các cột số phù hợp cho clustering
    if feature_cols is None:
        feature_cols = [
            'Age', 'Income', 'SpendingScore',
            'PurchaseFrequency', 'AverageOrderValue', 'LastPurchaseDays',
            'Gender_Encoded', 'MembershipLevel_Encoded',
            'PreferredCategory_Encoded'
        ]
        # Chỉ lấy các cột tồn tại trong DataFrame
        feature_cols = [col for col in feature_cols if col in df.columns]

    print(f"   - Các features: {feature_cols}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    print(f"   - Shape sau chuẩn hóa: {X_scaled.shape}")
    print(f"   ✅ Chuẩn hóa thành công!")

    return X_scaled, scaler, feature_cols


def preprocess_pipeline(df):
    """
    Pipeline tiền xử lý dữ liệu hoàn chỉnh.

    Thực hiện tuần tự:
    1. Xử lý missing values
    2. Xóa duplicates
    3. Label Encoding
    4. Chuẩn hóa dữ liệu

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame gốc

    Returns:
    --------
    dict
        Dictionary chứa:
        - 'df_cleaned': DataFrame đã tiền xử lý
        - 'X_scaled': Dữ liệu đã chuẩn hóa
        - 'scaler': Đối tượng StandardScaler
        - 'encoders': Dictionary các LabelEncoder
        - 'feature_cols': Danh sách tên features
    """
    print("=" * 60)
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

    # Bước 1: Xử lý missing values
    df = handle_missing_values(df)

    # Bước 2: Xóa duplicates
    df = remove_duplicates(df)

    # Bước 3: Label Encoding
    df, encoders = encode_categorical(df)

    # Bước 4: Chuẩn hóa dữ liệu
    X_scaled, scaler, feature_cols = scale_features(df)

    print("\n" + "=" * 60)
    print("✅ TIỀN XỬ LÝ HOÀN TẤT!")
    print(f"   - Kích thước cuối: {df.shape}")
    print("=" * 60)

    return {
        'df_cleaned': df,
        'X_scaled': X_scaled,
        'scaler': scaler,
        'encoders': encoders,
        'feature_cols': feature_cols
    }


# =============================================================================
# MAIN - Chạy trực tiếp để test tiền xử lý
# =============================================================================
if __name__ == '__main__':
    import os

    # Đọc dữ liệu
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'customers.csv')

    if not os.path.exists(data_path):
        print("⚠️ Chưa có file customers.csv!")
        print("   Hãy chạy generate_data.py trước.")
    else:
        df = load_data(data_path)
        result = preprocess_pipeline(df)

        # Hiển thị kết quả
        print("\n📊 Dữ liệu sau tiền xử lý:")
        print(result['df_cleaned'].head())
